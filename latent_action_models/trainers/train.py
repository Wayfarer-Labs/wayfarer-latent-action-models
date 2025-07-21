import  os
import  piq
import  time
import  torch
import  wandb
import  random
import  pathlib
import  dataclasses
import  einops as eo
from    pathlib     import Path
from    typing      import TypedDict, NotRequired
from    torch.types import Number
from    torch       import Tensor
import  torch.nn.functional as F

from latent_action_models.models.latent_action_model    import ActionEncodingInfo, LatentActionModel, LatentActionModelOutput
from latent_action_models.configs                       import LatentActionModelTrainingConfig
from latent_action_models.trainers.base_trainer         import BaseTrainer
from latent_action_models.action_creation.clustering    import umap_visualization
from latent_action_models.utils                         import as_wandb_video

class LogStats(TypedDict):
    loss:           float
    kl_loss:        float
    recon_loss:     float
    mu:             Tensor
    logvar:         Tensor
    grad_norm:      float
    learning_rate:  float
    weight_decay:   float
    iter_sec:       NotRequired[float]
    psnr:           NotRequired[Number]
    ssim:           NotRequired[Number]


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
    )

def kl_divergence(mean: Tensor, logvar: Tensor) -> Tensor:
    return -0.5 * torch.sum(1 + logvar - mean ** 2 - logvar.exp(), dim=1).mean()

class Trainer_LatentActionModel(BaseTrainer):
    def __init__(self, config: LatentActionModelTrainingConfig) -> None:
        latent_action_model = create_latent_action_model(config)
        super(Trainer_LatentActionModel, self).__init__(latent_action_model, config, device="cpu")
        self.beta           = 0.
        self._wandb_run     = None
        self.cfg: LatentActionModelTrainingConfig = config # -- reassign for typechecking:)

    @property
    def save_path(self) -> str: return f'lam_s{self.global_step}.pt'

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
        print(f"[rank-{self.rank}] checkpoint saved to {path}")


    def format_batch(self) -> Tensor:
        try:    video_bnchw = next(self.iter_loader)
        except  StopIteration: self.iter_loader = iter(self.dataloader) ; return self.format_batch()
        return  video_bnchw


    def train_step(self, video_bnchw: Tensor) -> LogStats:
        with self.amp_ctx():
            future_frame_video_bnchw             = video_bnchw[:, 1:]
            lam_outputs: LatentActionModelOutput = self.model(video_bnchw)
            reconstructed_frames_bnchw           = lam_outputs['reconstructed_video_bnchw']

            mse_loss    = F.mse_loss(reconstructed_frames_bnchw, future_frame_video_bnchw)
            kl_loss     = kl_divergence(lam_outputs['mean_bn1d'], lam_outputs['logvar_bn1d'])
            total_loss  = mse_loss + (self.beta * kl_loss)
            grad_norm   = self.optim_step(total_loss)
            
            if self.should_log:
                reconstructed_frames_nchw   = eo.rearrange(lam_outputs['reconstructed_video_bnchw'],
                                                            'b n c h w -> (b n) c h w')
                future_frame_video_nchw     = eo.rearrange(future_frame_video_bnchw,
                                                            'b n c h w -> (b n) c h w')
                psnr                        = piq.psnr(reconstructed_frames_nchw, future_frame_video_nchw).mean().item()
                ssim                        = piq.ssim(reconstructed_frames_nchw, future_frame_video_nchw).mean().item()

            return LogStats( loss           = total_loss    .item(),
                            kl_loss         = kl_loss       .item(), 
                            recon_loss      = mse_loss      .item(),
                            mu              = lam_outputs['mean_bn1d'],
                            logvar          = lam_outputs['logvar_bn1d'],
                            grad_norm       = grad_norm,
                            learning_rate   = self.optimizer.param_groups[0]["lr"],
                            weight_decay    = self.optimizer.param_groups[0]["weight_decay"],
                            psnr            = psnr if self.should_log else None,
                            ssim            = ssim if self.should_log else None)

    def log_step(self, stats: LogStats) -> None:
        # -- lazy init wandb
        if self._wandb_run is None:
            run_name        = self.cfg.run_name or f"LAM_{time.time():.0f}"
            self._wandb_run = wandb.init(project=self.cfg.wandb_project,
                                         name=run_name, config=dataclasses.asdict(self.cfg))

        mu_bn1d     = stats["mu"]
        logvar_bn1d = stats["logvar"]

        mu_mean     = mu_bn1d   .mean() .item()
        mu_std      = mu_bn1d   .std()  .item()
        sigma       = (0.5 * logvar_bn1d).exp()
        sigma_mean  = sigma     .mean() .item()

        wandb_dict = {
            # -- core losses
            "loss/total":           stats["loss"],
            "loss/recon":           stats["recon_loss"],
            "loss/kl":              stats["kl_loss"],
        
            # -- optimisation
            "optim/grad_norm":      stats["grad_norm"],
            "optim/lr":             stats["learning_rate"],
            "optim/weight_decay":   stats["weight_decay"],

            # -- latent health
            "latent/mu_mean":       mu_mean,
            "latent/mu_std":        mu_std,
            "latent/sigma_mean":    sigma_mean,

            # -- reconstruction metrics
            "reconstruction/psnr":  stats.get("psnr"),
            "reconstruction/ssim":  stats.get("ssim"),
        }

        if "iter_sec" in stats: wandb_dict["perf/iter_sec"] = stats["iter_sec"]

        wandb.log(wandb_dict, step=self.global_step)
        print(f"[rank {self.rank}] {wandb_dict}")


    def train(self)   -> None:
        while self.should_train:
            video_bnchw: Tensor   = self.format_batch()          ; start_time   = time.time()
            info:        LogStats = self.train_step(video_bnchw) ; iter_time    = time.time() - start_time
            info['iter_sec']      = iter_time / self.batch_size

            if self.should_log:      self.log_step(info)
            if self.should_save:     self.save_checkpoint(self.ckpt_dir / self.save_path)
            if self.should_validate: self.validate()
            self.global_step     += 1

        self.save_checkpoint(self.ckpt_dir / self.save_path)
        return


    def validate(self) -> None:
        num_processed_batches       = 0
        num_batches_umap            = self.cfg.val_num_samples_umap // self.batch_size
        # -- umap visualization
        latent_actions_list_bn1d    = []
        # -- reconstruction visualization. choose indices to keep for reconstruction
        num_recon_samples           = self.cfg.val_num_samples_recon
        recon_batch_idx: set[int]   = set(random.sample(range(num_batches_umap), k=num_recon_samples))
        recon_videos_list_bnchw     = []
        model: LatentActionModel    = self._model_unwrapped() # will this interfere with ddp??

        while num_processed_batches < num_batches_umap:
            video_bnchw: Tensor                     = self  .format_batch()
            action_info: ActionEncodingInfo         = model .encode_to_actions(video_bnchw)
            latent_actions_list_bn1d               += [action_info['mean_bn1d']]

            if num_processed_batches in recon_batch_idx:
                recon_videos_list_bnchw            += [video_bnchw]
            
            num_processed_batches                  += 1
        
        if self._wandb_run:
            # -- umap 
            latent_actions_bn1d             = torch.cat   (latent_actions_list_bn1d, dim=0)
            latent_actions_n1d              = eo.rearrange(latent_actions_bn1d, 'b n 1 d -> (b n) 1 d')

            umap_embed_n2, cluster_ids_n    = umap_visualization(latent_actions_n1d,
                                                                 vis_filename=f'umap_visualization_{self.save_path}'.replace('.pt', ''))

            table = wandb.Table(columns=['umap_x', 'umap_y', 'cluster'])
            for (x,y), c in zip(umap_embed_n2, cluster_ids_n):
                table.add_data(x,y, int(c))

            scatter = wandb.plot.scatter(table, "umap_x", "umap_y", title="UMAP Projection of Latent Actions")

            # -- reconstruction of image
            recon_videos_bnchw                      = torch.cat(recon_videos_list_bnchw, dim=0)
            lam_outputs: LatentActionModelOutput    = model.forward(recon_videos_bnchw)

            condition_video_bnchw:  Tensor          = lam_outputs["condition_video_bnchw"]
            recon_video_bnchw:      Tensor          = lam_outputs["reconstructed_video_bnchw"]
            gt_video_bnchw:         Tensor          = lam_outputs["groundtruth_video_bnchw"]
            
            video_table                             = wandb.Table(columns=["conditioning", "predicted", "ground_truth"])
            for cond, recon, gt in zip(condition_video_bnchw, recon_video_bnchw, gt_video_bnchw):
                table.add_data(
                    as_wandb_video(cond,  "Conditioning"),
                    as_wandb_video(recon, "Predicted next-frame"),
                    as_wandb_video(gt,    "Ground-truth next-frame"),
                )


            wandb.log({
                "UMAP Scatter": scatter,
                "Reconstruction": video_table,
            }, step=self.global_step)

if __name__ == "__main__":
    config = LatentActionModelTrainingConfig.from_yaml("configs/lam_training_tiny.yml")
    trainer = Trainer_LatentActionModel(config)
    trainer.train()