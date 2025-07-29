import  torch
import  einops      as eo
import  torch.nn    as nn
from    torch       import Tensor
import  torch.nn.functional as F
from    typing      import TypedDict, Literal, Any

from    latent_action_models.models.st_transformer import (
    S_Transformer,
    ST_Transformer,
    CrossAttention  # action conditioning  
)
import  latent_action_models.utils as utils


class ActionEncodingInfo(TypedDict):
    action_bn1d:        Tensor
    mean_bn1d:          Tensor
    logvar_bn1d:        Tensor

class ActionDecodingInfo(TypedDict):
    condition_video_bnchw:     Tensor
    reconstructed_video_bnchw: Tensor

class LatentActionModelOutput(TypedDict):
    # -- populated during inference: groundtruth next-state
    groundtruth_video_bnchw:    Tensor
    # -- encoding
    action_bn1d:                Tensor
    mean_bn1d:                  Tensor
    logvar_bn1d:                Tensor
    # -- reconstruction
    condition_video_bnchw:      Tensor
    reconstructed_video_bnchw:  Tensor


class LatentActionModel(nn.Module):
    def __init__(self,
                 video_dims:            tuple[int, int],
                 in_dim:                int,
                 model_dim:             int,
                 vae_dim:               int,
                 patch_size:            int,
                 num_enc_blocks:        int,
                 num_dec_blocks:        int,
                 num_heads:             int,
                 dropout:               float = 0.0,
                 conditioning:          Literal['add', 'crossattn'] = 'add',
                 conditioning_kwargs:   dict[str, Any]              = {}):

        super(LatentActionModel, self).__init__()
        self.in_dim     = in_dim
        self.model_dim  = model_dim
        self.vae_dim    = vae_dim
        self.patch_size = patch_size
        self.patch_token_dim = in_dim * (patch_size ** 2)

        self.video_height, self.video_width = video_dims

        self.action_prompt  = nn.Parameter(torch.empty(1,1,1, self.patch_token_dim))
        # -- how the video and actions get mixed together
        self.conditioning   = conditioning
        self.cond_kwargs    = conditioning_kwargs
        self.cond_net       = nn.Identity()
        
        if self.conditioning == 'crossattn':
            self.cond_net   = CrossAttention(num_heads  = (num_heads // 4) or 1,
                                             d_model    = self.model_dim,
                                             **self.cond_kwargs)
        
        nn.init.uniform_(self.action_prompt, a=-1, b=1)
    
        self.encoder        = ST_Transformer(in_dim=self.patch_token_dim,
                                             model_dim=self.model_dim,
                                             out_dim=self.model_dim,
                                             num_blocks=num_enc_blocks,
                                             num_heads=num_heads, dropout=dropout)
        
        # -- vae
        self.moments_proj   = nn.Linear(model_dim,              self.vae_dim * 2)
        self.patch_proj     = nn.Linear(self.patch_token_dim,   model_dim)
        self.action_proj    = nn.Linear(self.vae_dim,           model_dim)
        
        self.decoder        = S_Transformer(in_dim=self.model_dim,
                                            model_dim=model_dim,
                                            out_dim=self.patch_token_dim,
                                            num_blocks=num_dec_blocks,
                                            num_heads=num_heads,
                                            dropout=dropout)

        print(f'[LatentActionModel] model initialized with num_params={sum(p.numel() for p in self.parameters()):,}')
    
    
    def _patchify(self, video_bnchw: Tensor) -> Tensor:
        return utils.patchify(video_bnchw, size=self.patch_size)

    def _unpatchify(self, video_patches_bnpd: Tensor) -> Tensor:
        return utils.unpatchify(video_patches_bnpd, size=self.patch_size,
                                h_out=self.video_height, w_out=self.video_width)

    def encode_to_actions(self, video_bnchw: Tensor) -> ActionEncodingInfo:
        B,N, *_             = video_bnchw.shape
        video_patches_bnpd  = self._patchify(video_bnchw)
        actions_bn1d        = self.action_prompt.expand(B,N,-1,-1)
        # -- cat actions to video patches. p := p+1 here.
        patches_bnpd        = torch.cat([actions_bn1d, video_patches_bnpd], dim=2)  # -- patches_bnpd is centered herer with mu=0, var=1., 
        latents_bnpd        = self.encoder(patches_bnpd) # -- after the encoder, we get var of 700K. 

        # -- latent action is the zero-th token between each consecutive frame,
        # so n := n-1 here.
        latent_action_bnd   = latents_bnpd[:, 1:, 0]
        latent_actions_bd   = eo.rearrange(latent_action_bnd, 'b n d -> (b n) d')

        # -- variational autoencoder. v := 2 * vae_dim
        moments_bv          = self.moments_proj(latent_actions_bd)
        mean_bv, logvar_bv  = torch.chunk(moments_bv, 2, dim=1)
        
        action_bv           = mean_bv
        if self.training:
            action_bv       = action_bv + torch.randn_like(logvar_bv) * torch.exp(logvar_bv * 0.5)
        
        return ActionEncodingInfo(  action_bn1d  = eo.rearrange(action_bv,  '(b n) d -> b n 1 d', b=B),
                                    mean_bn1d    = eo.rearrange(mean_bv,    '(b n) d -> b n 1 d', b=B),
                                    logvar_bn1d  = eo.rearrange(logvar_bv,  '(b n) d -> b n 1 d', b=B))

    def condition_video_to_actions(self,
                                   video_patches_bpnd:  Tensor,
                                   action_proj_bn1c:    Tensor) -> Tensor:
        if self.conditioning == 'add':          return video_patches_bpnd + action_proj_bn1c
        if self.conditioning == 'crossattn':    return self.cond_net(action_proj_bn1c, video_patches_bpnd)

    def decode_to_frame(self,
                        video_bnchw: Tensor,
                        action_bn1d: Tensor) -> ActionDecodingInfo:
        # -- d: patch_token_dim. raw patchified video (not encoding) to prevent collusion between encoder/decoder
        # -- v: vae_dim (dimension of mixture of gaussians in vae)
        # -- c: model_dim, dimension of the pixel decoder (also identical to encoder dim)
        video_patches_bnpd                  = self._patchify(video_bnchw)
        prev_video_proj_patches_bnpc        = self.patch_proj(video_patches_bnpd)
        action_proj_patches_bn1c            = self.action_proj  (action_bn1d)

        action_conditioned_patches_bnpc     = self.condition_video_to_actions(prev_video_proj_patches_bnpc, action_proj_patches_bn1c)
        
        video_reconstruction_bnpd           = self.decoder(prev_video_proj_patches_bnpc + action_proj_patches_bn1c)
        video_reconstruction_bnpd           = F.sigmoid(video_reconstruction_bnpd)
        video_reconstruction_bnchw          = self._unpatchify(video_reconstruction_bnpd)

        return ActionDecodingInfo(condition_video_bnchw     = video_bnchw,
                                  reconstructed_video_bnchw = video_reconstruction_bnchw)

    def forward(self, video_bnchw: Tensor) -> LatentActionModelOutput:
        action_info:         ActionEncodingInfo = self.encode_to_actions(video_bnchw)
        groundtruth_video_bnchw                 = video_bnchw[:,:-1,::]
        reconstruction_info: ActionDecodingInfo = self.decode_to_frame  (groundtruth_video_bnchw,
                                                                         action_info['action_bn1d'])

        return LatentActionModelOutput(groundtruth_video_bnchw = groundtruth_video_bnchw,
                                        **action_info,
                                        **reconstruction_info)

if __name__ == '__main__':
    video_bnchw = torch.randn(4, 32, 3, 224, 224)
    lam = LatentActionModel(video_dims=(64,64), in_dim=3,
                            model_dim=64, vae_dim=16,
                            patch_size=8, num_enc_blocks=4,
                            num_dec_blocks=4, num_heads=4)
    out = lam(video_bnchw)
    print({k: v.shape for k,v in out.items()})
