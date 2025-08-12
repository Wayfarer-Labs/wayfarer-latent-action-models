import  os
import  piq
import  time
import  polars as pl
import  torch
import  pandas as pd
from torch._dynamo.eval_frame import null_context
from torch.utils.data import Dataset, IterableDataset
import  wandb
import  pathlib
import  traceback
from    contextlib import nullcontext
import  dataclasses
from    toolz import identity
import  einops as eo
import  plotly.express as px
from    torch.types import Number
from    torch       import Tensor

import  torch.nn.functional as F
import  plotly.figure_factory as ff
import  plotly.graph_objects as go
from    typing      import TypedDict, NotRequired, Literal, Generator, TypedDict, Any


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


class BatchEvent(TypedDict):
    video_vnchw:    Tensor
    meta:           dict[str, Any]
    data_sec:       float
    epoch:          int


def timed(iterator, *, sync_cuda: bool = False):
    it = iter(iterator)
    while True:
        t0 = time.perf_counter()
        try:
            batch = next(it)
            if sync_cuda and torch.cuda.is_available(): torch.cuda.synchronize()
            yield (
                (time.perf_counter() - t0),
                batch
            )
        except StopIteration: break


class LogStats(TypedDict):
    loss:           Tensor
    kl_loss:        Tensor
    recon_loss:     Tensor
    baseline_loss:  Tensor
    condition_loss: Tensor
    mu:             Tensor
    logvar:         Tensor
    beta:           float
    grad_norm:      float
    learning_rate:  float
    weight_decay:   float
    lam_outputs:    LatentActionModelOutput
    iter_sec:       NotRequired[float]
    batch_sec:      NotRequired[float]
    rho:            NotRequired[Tensor | None]
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
        conditioning    = config.conditioning,
        conditioning_kwargs = config.conditioning_kwargs,
        total_steps         = config.max_steps
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
        self.on_latents         = 'latent' in config.data_config.dataset_name
        self.decoder            = R3DCDecodingPipeline() if self.on_latents else None
        self.cfg: LatentActionModelTrainingConfig = config # -- reassign for typechecking:)
        self.loss_variant: Literal['residual', 'reconstruction'] = self.cfg.loss_variant
        self.conditioning_type: Literal['add', 'crossattn', 'gated_crossattn'] = self.cfg.conditioning

    @property
    def save_path(self) -> str: return f'lam_s{self.global_step}_beta{self.cfg.beta}_stride{self.cfg.data_config.stride}_vaedim{self.cfg.vae_dim}.pt'

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


    def _format_batch(self, batch: tuple) -> tuple[Tensor, dict]:
        video_bnchw, metadata   = batch
        video_bnchw             = video_bnchw.to(self.device, non_blocking=True)
        
        if not self.on_latents:
            video_bnchw = (video_bnchw + 1.) / 2.

        return (
            video_bnchw,
            metadata
        )


    def iter_dataset(
        self,
        long_batch: bool = False,
        *,
        sync_cuda_timing: bool = True,
    ) -> Generator[BatchEvent, None, None]:
        """
        - Map-style: ends on StopIteration (one pass). Trainer owns epoch setup.
        - Iterable: recreates iterator on StopIteration (infinite stream).
        - Measures per-batch data time (includes H2D if sync_cuda_timing=True).
        """
        loader      = self.long_dataloader if long_batch else self.dataloader
        it          = self.iter_long_loader if long_batch else self.iter_loader
        is_iterable = isinstance(self.dataset, IterableDataset)

        while True:
            t0 = time.perf_counter()
            try:
                batch = next(it)
            except StopIteration:
                if is_iterable:
                    # infinite stream semantics
                    it = iter(loader)
                    if long_batch: self.iter_long_loader = it
                    else:          self.iter_loader      = it
                    continue
                else:
                    # one pass per call (epoch ends here)
                    break
            except Exception as e:
                print(f'[rank {self.rank}] iter_dataset({long_batch=}) error: {e}')
                traceback.print_exc()
                continue

            # Move/normalize inside timing window
            video, meta = self._format_batch(batch)
            if sync_cuda_timing and torch.cuda.is_available():
                torch.cuda.synchronize()
            data_sec = time.perf_counter() - t0

            yield BatchEvent(video=video, meta=meta, data_sec=data_sec, epoch=self.epoch)


    def reconstruction_loss(self, lam_outputs: LatentActionModelOutput, as_baseline: bool = False, as_condition: bool = False) -> Tensor:
        # -- as_baseline:   get the mse between condition and target, as the loss when learning the identity function
        # -- as_condition:  get the mse between the condition and prediction, which shows how close we are to the condition.
        # very similar to as_baseline except one is an upper bound on meaningful loss and the other will tend to 0 if the condition is memorized
        assert not (as_baseline and as_condition)
        with torch.set_grad_enabled(not as_baseline and not as_condition):
            condition   = lam_outputs['condition_video_bnchw']
            prediction  = lam_outputs['reconstructed_video_bnchw'] if not as_baseline   else condition
            target      = lam_outputs['next_state_video_bnchw']    if not as_condition  else condition

            if self.loss_variant == 'residual':
                return F.mse_loss(prediction - condition, target - condition)
            if self.loss_variant == 'reconstruction':
                return F.mse_loss(prediction, target)
            
            raise NotImplementedError(f'{self.loss_variant=}')

    def train_step(self, video_bnchw: Tensor) -> LogStats:
        with self.amp_ctx():
            lam_outputs: LatentActionModelOutput = self.model(video_bnchw)

            # -- loss showing how bad we'd be if we memorize the conditioning
            baseline_loss   = self.reconstruction_loss(lam_outputs, as_baseline=True)
            condition_loss  = self.reconstruction_loss(lam_outputs, as_condition=True)
            mse_loss        = self.reconstruction_loss(lam_outputs, as_baseline=False, as_condition=False)
            kl_loss         = kl_divergence(lam_outputs['mean_bn1d'], lam_outputs['logvar_bn1d'])
            beta            = next(self.beta_scheduler)
            total_loss      = mse_loss + (beta * kl_loss)
            grad_norm       = self.optim_step(total_loss)

            if self.should_log:
                reconstructed_frames_nchw   = eo.rearrange(lam_outputs['reconstructed_video_bnchw'],
                                                            'b n c h w -> (b n) c h w').clip(0, 1)
                future_frame_video_nchw     = eo.rearrange(lam_outputs['next_state_video_bnchw'],
                                                            'b n c h w -> (b n) c h w').clip(0, 1)
                psnr                        = piq.psnr(reconstructed_frames_nchw, future_frame_video_nchw).mean().item()
                if self.on_latents: ssim    = None
                else:               ssim    = piq.ssim(reconstructed_frames_nchw, future_frame_video_nchw).mean().item()
            else: psnr = ssim = None

            return LogStats( loss            = total_loss,
                             kl_loss         = kl_loss, 
                             recon_loss      = mse_loss,
                             baseline_loss   = baseline_loss,
                             condition_loss  = condition_loss,
                             rho             = self.model.cond_net.rho if self.conditioning_type == 'gated_crossattn' else None,
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
        rho         = stats["rho"] and stats["rho"].detach().item()

        wandb_dict = {
            # -- core losses
            "loss/total":           stats["loss"].item(),
            "loss/recon":           stats["recon_loss"].item(), # prev to next
            "loss/kl":              stats["kl_loss"].item(),
            "loss/beta":            stats["beta"],
            "loss/next_to_prev":    stats["baseline_loss"].item(),
            "loss/pred_to_prev":    stats["condition_loss"].item(),
            "loss/rho":             rho,
        
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

    def log_debug_sample(self, video_bnchw: Tensor):
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

    def train(self):
        for self.epoch in range(self.num_epochs):
            if not self.is_iterable:
                if (hasattr(self.dataloader, "sampler") and 
                    hasattr(self.dataloader.sampler, "set_epoch")
                ):
                    self.dataloader.sampler.set_epoch(self.epoch)
                self.iter_loader = iter(self.dataloader)
            elif self.epoch == 0:
                self.iter_loader = iter(self.dataloader)

            for step_in_epoch, event in enumerate(self.iter_dataset(sync_cuda_timing=True)):
                batch_time          = event["data_sec"]
                video_bnchw         = event["video"]
                start_time          = time.perf_counter()
                info: LogStats      = self.train_step(video_bnchw) ; torch.cuda.synchronize()
                train_time          = time.perf_counter() - start_time

                info['iter_sec']    = batch_time + train_time
                info['batch_sec']   = batch_time
                info['train_sec']   = train_time

                if self.should_log:         self.log_step(info)
                if self.should_save:        self.save_checkpoint(self.ckpt_dir / self.save_path)
                if self.should_validate:    self.validate_simple()

                self.global_step += 1 ; self.commit_log()
                self.model.bump_step()
                if self.rank == 0: print(f'[rank {self.rank}] step {self.global_step}')
                # -- in the iterable dataset case, we just have 1 epoch of size max_steps
                if (step_in_epoch+1) >= self.epoch_steps:
                    break
        
        if self.rank == 0: print(f'[rank {self.rank}] training complete.')

    @torch.no_grad()
    def validate_simple(self) -> None:
        model: LatentActionModel = self._model_unwrapped()

        # ---- targets ----
        target_umap   = int(self.cfg.val_num_samples_umap)
        target_recon  = int(self.cfg.val_num_samples_recon)

        # ---- collectors ----
        latent_actions_list_bn1d: list[Tensor] = []
        recon_videos_list_bnchw: list[Tensor]  = []

        # ---- iterators (normal + long) ----
        ev_iter      = self.iter_dataset(sync_cuda_timing=False)
        ev_iter_long = self.iter_dataset(long_batch=True, sync_cuda_timing=False)

        # ---- pull until quotas met ----
        umap_done = recon_done = False
        umap_count = recon_count = 0

        while not (umap_done and recon_done):
            # get a standard batch event; if exhausted (map-style), recreate iterator and continue
            try:
                ev = next(ev_iter)
            except StopIteration:
                # Map-style validation during training will just recreate from current dataloader
                if not self.is_iterable:
                    self.iter_loader = iter(self.dataloader)
                ev_iter = self.iter_dataset(sync_cuda_timing=False)
                ev = next(ev_iter)

            video_bnchw = ev["video"]

            if not umap_done:
                action_info = model.encode_to_actions(video_bnchw)
                latent_actions_list_bn1d.append(action_info["mean_bn1d"])
                # count = B * N (batch x temporal windows)
                umap_count += action_info["mean_bn1d"].shape[0] * action_info["mean_bn1d"].shape[1]
                umap_done   = umap_count >= target_umap

            if not recon_done:
                # take up to what we still need
                need = max(0, target_recon - recon_count)
                if need > 0:
                    recon_videos_list_bnchw.append(video_bnchw[:need])
                    recon_count += min(need, video_bnchw.shape[0])
                recon_done = recon_count >= target_recon

        print(f"[rank {self.rank}] validation data collected")

        if self._wandb_enabled:
            # -- UMAP prep (gather across ranks first) --
            latent_actions_bn1d = torch.cat(latent_actions_list_bn1d, dim=0)  # [B, N, 1, D]
            latent_actions_n1d  = eo.rearrange(latent_actions_bn1d, "b n 1 d -> (b n) 1 d")
            latent_actions_n1d  = gather_to_rank(latent_actions_n1d, dst=0, dim=0, cat=True)

            # -- Recon pass on the recon sample batch --
            recon_videos_bnchw    = torch.cat(recon_videos_list_bnchw, dim=0)  # [R, N, C, H, W]
            lam_outputs           = model.forward(recon_videos_bnchw.float())
            condition_video_bnchw = gather_to_rank(lam_outputs["condition_video_bnchw"],     dst=0, dim=0, cat=True)
            recon_video_bnchw     = gather_to_rank(lam_outputs["reconstructed_video_bnchw"], dst=0, dim=0, cat=True)
            gt_video_bnchw        = gather_to_rank(lam_outputs["next_state_video_bnchw"],    dst=0, dim=0, cat=True)

            # Residual variant: shift prediction once (not twice)
            if self.loss_variant == "residual":
                recon_video_bnchw = recon_video_bnchw + condition_video_bnchw

            # ---- Rollouts need long sequences (>2 frames) ----
            # Get a long-batch sample; if exhausted, recreate iterator and pull again
            try:
                ev_long = next(ev_iter_long)
            except StopIteration:
                if not self.is_iterable:
                    self.iter_long_loader = iter(self.long_dataloader)
                ev_iter_long = self.iter_dataset(long_batch=True, sync_cuda_timing=False)
                ev_long = next(ev_iter_long)

            rollout_video_GT_bnchw = ev_long["video"][0:1].float()  # take one sequence

            rollout_video_TF_nchw = self._model_unwrapped().autoregressive_rollout(
                rollout_video_GT_bnchw[0],
                teacher_forced=True,
                adjust_residual=(self.loss_variant == "residual"),
            )
            rollout_video_AR_nchw = self._model_unwrapped().autoregressive_rollout(
                rollout_video_GT_bnchw[0],
                teacher_forced=False,
                adjust_residual=(self.loss_variant == "residual"),
            )

            # ---- Decode to RGB if operating in latent space ----
            if self.on_latents:
                # flatten BN to (B*N) for bridge, then reshape back
                cond_nchw = eo.rearrange(condition_video_bnchw, "b n c h w -> (b n) c h w").bfloat16()
                pred_nchw = eo.rearrange(recon_video_bnchw,     "b n c h w -> (b n) c h w").bfloat16()
                gt_nchw   = eo.rearrange(gt_video_bnchw,        "b n c h w -> (b n) c h w").bfloat16()

                condition_video_nchw = F.sigmoid(self.decoder(cond_nchw))
                recon_video_nchw     = F.sigmoid(self.decoder(pred_nchw))
                gt_video_nchw        = F.sigmoid(self.decoder(gt_nchw))

                condition_video_bnchw = eo.rearrange(
                    condition_video_nchw, "(b n) c h w -> b n c h w", b=condition_video_bnchw.shape[0]
                )
                recon_video_bnchw = eo.rearrange(
                    recon_video_nchw, "(b n) c h w -> b n c h w", b=recon_video_bnchw.shape[0]
                )
                gt_video_bnchw = eo.rearrange(
                    gt_video_nchw, "(b n) c h w -> b n c h w", b=gt_video_bnchw.shape[0]
                )

                # also decode long sequences for rollouts
                gt_rollout_nchw = F.sigmoid(self.decoder(eo.rearrange(rollout_video_GT_bnchw, "b n c h w -> (b n) c h w").bfloat16()))
                tf_rollout_nchw = F.sigmoid(self.decoder(rollout_video_TF_nchw.bfloat16()))
                ar_rollout_nchw = F.sigmoid(self.decoder(rollout_video_AR_nchw.bfloat16()))
            else:
                # already pixels
                gt_rollout_nchw = eo.rearrange(rollout_video_GT_bnchw, "b n c h w -> (b n) c h w")
                tf_rollout_nchw = rollout_video_TF_nchw
                ar_rollout_nchw = rollout_video_AR_nchw

            if self.rank == 0:
                # --- UMAP scatter ---
                umap_embed_n2, _ = umap_visualization(latent_actions_n1d)
                umap_table = wandb.Table(columns=["umap_x", "umap_y"])
                for x, y in umap_embed_n2:
                    umap_table.add_data(x, y)
                scatter_plot = wandb.plot.scatter(umap_table, "umap_x", "umap_y",
                                                title=f"UMAP Visualization (Step {self.global_step})")

                # --- Recon table ---
                video_table = wandb.Table(columns=["cond | pred next | next"])
                for cond, recon, gt in zip(condition_video_bnchw, recon_video_bnchw, gt_video_bnchw):
                    frame_cond_recon_gt = torch.cat([cond, recon, gt], dim=-1)
                    video_table.add_data(as_wandb_video(frame_cond_recon_gt, "Conditioning | Predicted next-frame | Next-frame"))

                wandb.log({
                    f"UMAP Scatter Plot/{self.global_step}":            scatter_plot,
                    f"Reconstruction/{self.global_step}":               video_table,
                    f"Rollout/Groundtruth    - {self.global_step}":     as_wandb_video(gt_rollout_nchw),
                    f"Rollout/Teacher.forced - {self.global_step}":     as_wandb_video(tf_rollout_nchw),
                    f"Rollout/Autoregressive - {self.global_step}":     as_wandb_video(ar_rollout_nchw),
                }, step=self.global_step)

        print(f"[rank {self.rank}] validation done")
        barrier()



if __name__ == "__main__":
    config = LatentActionModelTrainingConfig.from_yaml("configs/lam_latent.yml")
    trainer = Trainer_LatentActionModel(config)
    trainer.train()
