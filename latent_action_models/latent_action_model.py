import torch
import torch.nn as nn
import torch.nn.functional as F

from latent_action_models.st_transformer import S_Transformer, ST_Transformer


class LatentActionModel(nn.Module):
    def __init__(self,
                 in_dim:            int,
                 model_dim:         int,
                 vae_dim:           int,
                 patch_size:        int,
                 num_enc_blocks:    int,
                 num_dec_blocks:    int,
                 num_heads:         int,
                 dropout:           float = 0.0):
        super(LatentActionModel, self).__init__()
        self.in_dim = in_dim
        self.model_dim = model_dim
        self.vae_dim = vae_dim
        self.patch_size = patch_size
        self.patch_token_dim = in_dim * (patch_size ** 2)

        self.action_prompt = nn.Parameter(torch.empty(1,1,1, self.patch_token_dim))
        nn.init.uniform_(self.action_prompt, a=-1, b=1)

        self.encoder = ST_Transformer(in_dim=self.patch_token_dim,
                                      model_dim=self.model_dim,
                                      out_dim=self.model_dim,
                                      num_blocks=num_enc_blocks,
                                      num_heads=num_heads, dropout=dropout)
        

        self.decoder = S_Transformer(in_dim=self.model_dim,
                                     model_dim=model_dim,
                                     out_dim=self.patch_token_dim,
                                     num_blocks=num_dec_blocks,
                                     num_heads=num_heads,
                                     dropout=dropout)