import  os
import  piq
import  time
import  polars as pl
import  torch
import  pandas as pd
import  wandb
import  pathlib
import  traceback
import  dataclasses
from    toolz import identity
import  einops as eo
import  plotly.express as px
from    torch.types import Number
from    torch       import Tensor
import  torch.nn.functional as F
import  plotly.figure_factory as ff
import  plotly.graph_objects as go
from    typing      import TypedDict, NotRequired

from latent_action_models.datasets.video_loader import video_collate_fn
from latent_action_models.models.latent_action_model    import ActionEncodingInfo, LatentActionModel, LatentActionModelOutput
from latent_action_models.configs                       import LatentActionModelTrainingConfig
from latent_action_models.trainers.base_trainer         import BaseTrainer
from latent_action_models.action_creation.clustering    import umap_visualization
from latent_action_models.utils                         import (
    as_wandb_video, barrier, gather_to_rank,
    init_distributed, colors_labels_from_actions,
    get_quiver_vectors_from_actions, gather_objects_to_rank
)
from latent_action_models.data_exploration              import create_actions_parquet as lam_parquet
from latent_action_models.vae_bridge.owl_vae_bridge     import R3DCDecodingPipeline 


class LogStats(TypedDict):
    loss:           Tensor
    kl_loss:        Tensor
    recon_loss:     Tensor
    baseline_loss:  Tensor
    mu:             Tensor
    logvar:         Tensor
    beta:           float
    grad_norm:      float
    learning_rate:  float
    weight_decay:   float
    lam_outputs:    LatentActionModelOutput
    iter_sec:       NotRequired[float]
    batch_sec:      NotRequired[float]
    psnr:           NotRequired[Number | None]
    ssim:           NotRequired[Number | None]

def create_latent_action_model(config: LatentActionModelTrainingConfig) -> LatentActionModel:
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
        conditioning    = config.conditioning
    )


def kl_divergence(mean_bn1d: Tensor, logvar_bn1d: Tensor) -> Tensor:
    kl_elem_bn1d = 0.5 * (mean_bn1d.pow(2) + logvar_bn1d.exp() - 1.0 - logvar_bn1d)
    kl_total_b1  = kl_elem_bn1d.sum(dim=(1,3))
    return kl_total_b1.mean()


class Trainer_LatentActionModel(BaseTrainer):
    def __init__(self, config: LatentActionModelTrainingConfig) -> None:
        latent_action_model = create_latent_action_model(config)
        *_, device = init_distributed()
        super(Trainer_LatentActionModel, self).__init__(latent_action_model, config, device=device)
        self.beta_scheduler     = config.beta_scheduler_linear_warmup()
        self._wandb_run         = None
        self.debug_show_samples = 0
        self.on_latents         = config.data_config.dataset_name == 'owl_data_latent'
        self.decoder            = R3DCDecodingPipeline() if self.on_latents else None
        self.cfg: LatentActionModelTrainingConfig = config # -- reassign for typechecking:)

    @property
    def save_path(self) -> str: return f'lam_s{self.global_step}_beta{self.cfg.beta}_stride{self.cfg.data_config.stride}.pt'

    def load_checkpoint(self, path: os.PathLike) -> None:
        ckpt = torch.load(path, map_location="cpu")
        self._model_unwrapped().load_state_dict(ckpt["model"], strict=True)

        self.optimizer.load_state_dict(ckpt["optimizer"])
        self.scheduler.load_state_dict(ckpt["scheduler"])

        self.global_step = ckpt.get("step", 0)
        self.cfg         = self.cfg.from_dict(ckpt['cfg'])

        torch       .set_rng_state    (ckpt['torch_rng'])
        torch.cuda  .set_rng_state_all(ckpt['cuda_rng']) if ckpt['cuda_rng'] is not None else ()
        print(f"[rank {self.rank}] resumed from {path} @ step {self.global_step}")


    def save_checkpoint(self, path: os.PathLike) -> None:
        path = pathlib.Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        torch.save(
            {
                "step":       self.global_step,
                "model":      self._model_unwrapped().state_dict(),
                "optimizer":  self.optimizer.state_dict(),
                "scheduler":  self.scheduler.state_dict(),
                "cfg":        dataclasses.asdict(self.cfg),
                "torch_rng":  torch     .get_rng_state(),
                "cuda_rng":   torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            },
            path,
        )
        print(f"[rank {self.rank}] checkpoint saved to {path}")


    def format_batch(self) -> tuple[Tensor, dict]:
        try:
            video_bnchw, metadata = next(self.iter_loader)
        except  StopIteration: self.iter_loader         = iter(self.dataloader) ; return self.format_batch()
        except  Exception as e:
            print(f"[rank {self.rank}] error in format_batch: {e}") ; traceback.print_exc()
            return self.format_batch()
        video_bnchw = video_bnchw.to(self.device) 
        if self.on_latents:
            video_bnchw = video_bnchw
        else:
            video_bnchw = (video_bnchw + 1.) / 2. # -1,1 to 0, 1

        return  (
            video_bnchw,
            metadata
        )


    def train_step(self, video_bnchw: Tensor) -> LogStats:
        with self.amp_ctx():
            lam_outputs: LatentActionModelOutput = self.model(video_bnchw)
            future_frame_video_bnchw             = lam_outputs['next_state_video_bnchw']
            reconstructed_frames_bnchw           = lam_outputs['reconstructed_video_bnchw']
            condition_video_bnchw                = lam_outputs['condition_video_bnchw']

            # -- loss showing how bad we'd be if we memorize the conditioning
            with torch.no_grad():
                baseline_loss = F.mse_loss(condition_video_bnchw, future_frame_video_bnchw)

            mse_loss        = F.mse_loss(reconstructed_frames_bnchw, future_frame_video_bnchw)
            kl_loss         = kl_divergence(lam_outputs['mean_bn1d'], lam_outputs['logvar_bn1d'])
            beta            = next(self.beta_scheduler)
            total_loss      = mse_loss + (beta * kl_loss)
            grad_norm       = self.optim_step(total_loss)

            if self.should_log:
                reconstructed_frames_nchw   = eo.rearrange(lam_outputs['reconstructed_video_bnchw'],
                                                            'b n c h w -> (b n) c h w').clip(0, 1)
                future_frame_video_nchw     = eo.rearrange(future_frame_video_bnchw,
                                                            'b n c h w -> (b n) c h w').clip(0, 1)
                psnr                        = piq.psnr(reconstructed_frames_nchw, future_frame_video_nchw).mean().item()
                if self.on_latents: ssim    = None
                else:               ssim    = piq.ssim(reconstructed_frames_nchw, future_frame_video_nchw).mean().item()
            else: psnr = ssim = None

            return LogStats( loss            = total_loss,
                             kl_loss         = kl_loss, 
                             recon_loss      = mse_loss,
                             baseline_loss   = baseline_loss,
                             mu              = lam_outputs['mean_bn1d'],
                             logvar          = lam_outputs['logvar_bn1d'],
                             beta            = beta,
                             grad_norm       = grad_norm,
                             learning_rate   = self.optimizer.param_groups[0]["lr"],
                             weight_decay    = self.optimizer.param_groups[0]["weight_decay"],
                             psnr            = psnr if self.should_log else None,
                             ssim            = ssim if self.should_log else None,
                             lam_outputs     = lam_outputs )


    def log_step(self, stats: LogStats) -> None:
        # -- lazy init wandb
        if self._wandb_run is None:
            run_name        = self.cfg.run_name or f"LAM_{time.time():.0f}"
            self._wandb_run = wandb.init(project=self.cfg.wandb_project,
                                         name=run_name, config=dataclasses.asdict(self.cfg))
            wandb.watch(self.model, log='gradients', log_freq=self.cfg.log_every * 20)
            print(f'[rank {self.rank}] wandb initialized, watching gradients...')
    
        mu_bn1d     = stats["mu"]
        logvar_bn1d = stats["logvar"]

        mu_mean     = mu_bn1d   .mean() .item()
        mu_std      = mu_bn1d   .std()  .item()
        sigma       = (0.5 * logvar_bn1d).exp()
        sigma_mean  = sigma     .mean() .item()

        wandb_dict = {
            # -- core losses
            "loss/total":           stats["loss"].item(),
            "loss/recon":           stats["recon_loss"].item(),
            "loss/kl":              stats["kl_loss"].item(),
            "loss/beta":            stats["beta"],
        
            # -- optimisation
            "optim/grad_norm":      stats["grad_norm"],
            "optim/lr":             stats["learning_rate"],
            "optim/weight_decay":   stats["weight_decay"],

            # -- latent health
            "latent/mu_mean":       mu_mean,
            "latent/mu_std":        mu_std,
            "latent/sigma_mean":    sigma_mean,

            # -- reconstruction metrics 
            "reconstruction-stats/psnr":   stats.get("psnr"),
            "reconstruction-stats/ssim":   stats.get("ssim"),
        }

        if "iter_sec" in stats:     wandb_dict["perf/iter_sec"]     = stats["iter_sec"]
        if "batch_sec" in stats:    wandb_dict["perf/batch_sec"]    = stats["batch_sec"]

        wandb.log(wandb_dict, step=self.global_step, commit=False)
        print(f"[rank {self.rank}] {wandb_dict}")

    def commit_log(self) -> None:
        if self._wandb_run: wandb.log({}, commit=True)

    def train(self)   -> None:
        while self.should_train and (start_time := time.time()):
            barrier()
            print(f"[rank {self.rank}] training step {self.global_step}")
            video_bnchw, *_       =  self.format_batch()          ; batch_time   = time.time() - start_time
            info:        LogStats =  self.train_step(video_bnchw) ; iter_time    = time.time() - start_time
            print(f"[rank {self.rank}] batch dims {video_bnchw.shape} - batch time {batch_time} train time {iter_time} - batch checksum {video_bnchw.sum()}")
            info['iter_sec']      =  iter_time / self.batch_size
            info['batch_sec']     =  batch_time

            if self.should_log:      self.log_step(info)
            if self.should_save:     self.save_checkpoint(self.ckpt_dir / self.save_path)
            if self.should_validate: self.validate_simple()
                    
            if bool(self.debug_show_samples) and self.should_log:
                if self.on_latents:
                    num_debug_samples = 3
                    video_nchw  = eo.rearrange(video_bnchw[:num_debug_samples], 'b n c h w -> (b n) c h w')
                    decoded     = self.decoder(video_nchw)
                    video_bnchw = eo.rearrange(decoded, '(b n) c h w -> b n c h w', b=num_debug_samples)
                    video_bnchw = F.sigmoid(video_bnchw)
                    
                wandb.log({
                    f'debug/sample_{self.debug_show_samples}_video': as_wandb_video(video_bnchw, "video"),
                }, step=self.global_step)
                self.debug_show_samples -= 1
            
            self.global_step     += 1 ; self.commit_log() # -- only commit when all logs are written so we keep global-step monotonic invariant
            print(f"[rank {self.rank}] training step done {self.global_step}")

        self.save_checkpoint(self.ckpt_dir / self.save_path)
        return

    @torch.no_grad()
    def validate_simple(self) -> None:
        model: LatentActionModel    = self._model_unwrapped()
        num_samples_umap            = self.cfg.val_num_samples_umap
        num_samples_recon           = self.cfg.val_num_samples_recon

        # -- Simplified data collection --
        num_processed_umap_samples  = 0
        _more_umap = lambda: num_processed_umap_samples < num_samples_umap
        num_processed_recon_samples = 0
        _more_recon = lambda: num_processed_recon_samples < num_samples_recon
        
        latent_actions_list_bn1d = []
        recon_videos_list_bnchw = []

        while _more_umap() or _more_recon():
            # REMOVED: No longer need paths or start indices from the batch
            video_bnchw, _ = self.format_batch()
            
            if _more_umap():
                action_info = model.encode_to_actions(video_bnchw)
                latent_actions_list_bn1d.append(action_info['mean_bn1d'])
                num_umap_samples_in_batch = action_info['mean_bn1d'].shape[0] * action_info['mean_bn1d'].shape[1]
                num_processed_umap_samples += num_umap_samples_in_batch

            if _more_recon():
                recon_videos_list_bnchw.append(video_bnchw[:num_samples_recon, ::])
                num_recon_samples_in_batch = video_bnchw.shape[0]
                num_processed_recon_samples += num_recon_samples_in_batch

        print(f"[rank {self.rank}] validation data collected")

        if self._wandb_enabled:
            # -- Gather tensors from all GPUs --
            latent_actions_bn1d     = torch.cat(latent_actions_list_bn1d, dim=0)
            latent_actions_n1d      = eo.rearrange(latent_actions_bn1d, 'b n 1 d -> (b n) 1 d')
            latent_actions_n1d      = gather_to_rank(latent_actions_n1d, dst=0, dim=0, cat=True)

            recon_videos_bnchw      = torch.cat(recon_videos_list_bnchw, dim=0)
            lam_outputs             = model.forward(recon_videos_bnchw.float())  # why do we cast here? idfk
            condition_video_bnchw   = gather_to_rank(lam_outputs["condition_video_bnchw"], dst=0, dim=0, cat=True)
            recon_video_bnchw       = gather_to_rank(lam_outputs["reconstructed_video_bnchw"], dst=0, dim=0, cat=True)
            gt_video_bnchw          = gather_to_rank(lam_outputs["next_state_video_bnchw"], dst=0, dim=0, cat=True)

            if self.rank == 0:
                # --- SIMPLIFIED UMAP PLOTTING ---
                
                # 1. Get the 2D UMAP embedding (assuming a simplified umap_visualization helper)
                umap_embed_n2, _ = umap_visualization(latent_actions_n1d)

                # 2. Create a wandb.Table with just the coordinates
                umap_table = wandb.Table(columns=['umap_x', 'umap_y'])
                for x, y in umap_embed_n2:
                    umap_table.add_data(x, y)
                
                # 3. Create a scatter plot object from the table
                scatter_plot = wandb.plot.scatter(
                    umap_table, 'umap_x', 'umap_y', title=f"UMAP Visualization (Step {self.global_step})"
                )

                # --- Reconstruction table (unchanged) ---
                video_table = wandb.Table(columns=["cond | pred next | next"])
                # -- decode into pixels if we are using latents
                if self.on_latents:
                    condition_video_nchw    = F.sigmoid(self.decoder(eo.rearrange(condition_video_bnchw,   'b n c h w -> (b n) c h w').bfloat16()))
                    recon_video_nchw        = F.sigmoid(self.decoder(eo.rearrange(recon_video_bnchw,       'b n c h w -> (b n) c h w').bfloat16()))
                    gt_video_nchw           = F.sigmoid(self.decoder(eo.rearrange(gt_video_bnchw,          'b n c h w -> (b n) c h w').bfloat16()))
                    condition_video_bnchw   = eo.rearrange(condition_video_nchw,                           '(b n) c h w -> b n c h w', b=num_samples_recon)
                    recon_video_bnchw       = eo.rearrange(recon_video_nchw,                               '(b n) c h w -> b n c h w', b=num_samples_recon)
                    gt_video_bnchw          = eo.rearrange(gt_video_nchw,                                  '(b n) c h w -> b n c h w', b=num_samples_recon)
            
                # -- add reconstructions 
                for cond, recon, gt in zip(condition_video_bnchw, recon_video_bnchw, gt_video_bnchw):
                    frame_cond_recon_gt = torch.cat([cond, recon, gt], dim=-1)
                    video_table.add_data(as_wandb_video(frame_cond_recon_gt, "Conditioning | Predicted next-frame | Next-frame"))

                # --- Simplified logging ---
                if self._wandb_run:
                    print(f"[rank {self.rank}] logging to wandb")
                    wandb.log({
                        f"UMAP Scatter Plot/{self.global_step}": scatter_plot,
                        f"Reconstruction/{self.global_step}": video_table,
                    }, step=self.global_step)

        print(f"[rank {self.rank}] validation done")
        barrier()


if __name__ == "__main__":
    config = LatentActionModelTrainingConfig.from_yaml("configs/lam_latent.yml")
    trainer = Trainer_LatentActionModel(config)
    trainer.train()
