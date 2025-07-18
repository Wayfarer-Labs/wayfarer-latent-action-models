import  os
import  time
import  torch
import  pathlib
from    typing  import TypedDict
from    torch   import nn, Tensor
import torch.nn.functional as F

from latent_action_models.models.latent_action_model import LatentActionModel, LatentActionModelOutput, lam
from latent_action_models.config                     import LatentActionModelConfig
CKPT_DIR = pathlib.Path('checkpoints/')


class LogStats(TypedDict, total=False):
    # TODO Loss, and what else?
    loss:           float
    kl_loss:        float
    recon_loss:     float
    mu:             float
    logvar:         float
    grad_norm:      float
    learning_rate:  float
    weight_decay:   float
    iter_sec:       float


def create_latent_action_model(config: LatentActionModelConfig) -> LatentActionModel:
    return LatentActionModel(
        video_dims      = config.video_dims,
        in_dim          = config.in_dim,
        model_dim       = config.model_dim,
        vae_dim         = config.vae_dim,
        patch_size      = config.patch_size,
        num_enc_blocks  = config.num_enc_blocks,
        num_dec_blocks  = config.num_dec_blocks,
        num_heads       = config.num_heads,
        dropout         = config.dropout,
    )


class Trainer_LatentActionModel(nn.Module):
    def __init__(self, config: LatentActionModelConfig) -> None:
        super().__init__(LatentActionModelConfig)
        self.config              = config
        self.global_step         = 0
        self.ckpt_dir            = CKPT_DIR
        # -- data
        self.batch_size         = config.batch_size
        # -- optim
        self.optimizer           = None
        self.scheduler           = None
        self.max_grad_norm       = None
        self.latent_action_model = create_latent_action_model(config)

        if self.should_load:       self.load_checkpoint()

    @property
    def should_log(self)   -> bool:
        pass

    @property
    def should_train(self) -> bool:
        pass

    @property
    def should_save(self)  -> bool:
        pass

    @property
    def should_load(self)  -> bool:
        pass

    @property
    def save_path(self) -> str: return f'lam_s{self.global_step}.pt'

    def load_checkpoint(self, path: os.PathLike) -> None:
        pass

    def save_checkpoint(self, path: os.PathLike) -> None:
        pass

    def format_batch(self) -> Tensor:
        pass

    def compute_loss(self) -> Tensor:
        pass

    def train_step(self, video_bnchw: Tensor) -> LogStats:
        future_frame_video_bnchw             = video_bnchw[:, 1:]
        lam_outputs: LatentActionModelOutput = self.latent_action_model.forward(video_bnchw)
        reconstructed_frames_bnchw           = lam_outputs['reconstructed_video_bnchw']

        mse_loss = F.mse_loss(reconstructed_frames_bnchw, future_frame_video_bnchw)
        kl_loss  = F.kl_div()

    def log_step(self, log_stats: LogStats) -> None:
        pass

    def optim_step(self, loss: Tensor) -> None:
        pass

    def train(self)   -> Tensor:
        while self.should_train:
            video_bnchw: Tensor   = self.format_batch()          ; start_time   = time.time()
            info:        LogStats = self.train_step(video_bnchw) ; iter_time    = time.time() - start_time
            if self.should_log:     self.log_step(info | {'iter_sec': iter_time / self.batch_size})
            if self.should_save:    self.save_checkpoint(self.ckpt_dir / self.save_path)
            self.global_step     += 1
