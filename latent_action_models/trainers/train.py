import  os
import  piq
import  time
import  polars as pl
import  shutil
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
from    typing      import TypedDict, NotRequired, Literal, Generator, TypedDict, Any, Optional

from latent_action_models.trainers.probe_trainer        import RegressionProbeTrainer
from latent_action_models.datasets.video_loader         import video_collate_fn
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



def kl_divergence(mean_bn1d: Tensor, logvar_bn1d: Tensor) -> Tensor:
    kl_elem_bn1d = 0.5 * (mean_bn1d.pow(2) + logvar_bn1d.exp() - 1.0 - logvar_bn1d)
    kl_total_b1  = kl_elem_bn1d.sum(dim=(1,3))
    return kl_total_b1.mean()


class Trainer_LatentActionModel(BaseTrainer):
    def __init__(self, config: LatentActionModelTrainingConfig) -> None:
        latent_action_model = LatentActionModel.from_config(config)
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
        self.setup_run_dir()

    def setup_run_dir(self) -> None:
        self.run_dir = self.ckpt_root / self.wandb_run_name
        self.run_dir.mkdir(parents=True, exist_ok=True)
        # -- copy yaml to the dir
        if self.cfg.config_path is not None:
            shutil.copy(self.cfg.config_path, self.run_dir / 'config.yml')

        self.ckpt_dir = self.run_dir / 'checkpoints'
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)


    @property
    def save_path(self) -> str: 
        return f'checkpoint_step={self.global_step}.pt'
        # return f'lam_s{self.global_step}_e{self.cfg.num_enc_blocks}-d{self.cfg.num_dec_blocks}_beta{self.cfg.beta}_stride{self.cfg.data_config.stride}_vaedim{self.cfg.vae_dim}.pt'

    @property
    def should_probe(self) -> bool: return self.cfg.probe_config and self.global_step % self.cfg.probe_every == 0

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
        validation: bool = False,
        *,
        sync_cuda_timing: bool = True,
    ) -> Generator[BatchEvent, None, None]:
        """
        - Map-style: ends on StopIteration (one pass). Trainer owns epoch setup.
        - Iterable: recreates iterator on StopIteration (infinite stream).
        - Measures per-batch data time (includes H2D if sync_cuda_timing=True).
        """
        loader = self.dataloader
        it = self.iter_loader
        if long_batch:
            loader = self.long_dataloader
            it = self.iter_long_loader
        if validation:
            loader = self.val_dataloader
            it = self.iter_val_loader
        
        is_iterable = isinstance(loader.dataset, IterableDataset)

        while True:
            t0 = time.perf_counter()
            try: batch = next(it)
            except StopIteration:
                if is_iterable:
                    # infinite stream semantics
                    it = iter(loader)
                    if long_batch: self.iter_long_loader = it
                    elif validation: self.iter_val_loader = it
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
            run_name        = self.cfg.run_name or self.cfg.wandb_run_name()
            self._wandb_run = wandb.init(project=self.cfg.wandb_project,
                                         name=run_name, config=dataclasses.asdict(self.cfg))
            wandb.define_metric("global_step")
            wandb.define_metric("*", step_metric="global_step")  # make all series use global_step
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

        wandb.log({"global_step": self.global_step, **wandb_dict}, commit=False)
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
            "global_step": self.global_step,
        })

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
                # TODO Split into validate_rollouts, validate_umap, and validate_action_controllability?
                if self.should_evaluate:    self.evaluate()

                self.global_step += 1 ; self.commit_log()
                self.model.bump_step()
                if self.rank == 0: print(f'[rank {self.rank}] step {self.global_step}')
                # -- in the iterable dataset case, we just have 1 epoch of size max_steps
                if (step_in_epoch+1) >= self.epoch_steps:
                    break
        
        if self.rank == 0: print(f'[rank {self.rank}] training complete.')

    @torch.no_grad()
    def validate_action_controllability(
        self,
        video_bnchw: Tensor,
        *,
        alphas: tuple[float, ...] = (0.0, 0.5, 1.0, 1.5, 2.0),
        max_eval_windows: int = 64,
        log_key: str = "controllability",
    ) -> dict[str, Any] | None:
        """
        Rank-0-only evaluation: scale the same latent action by α and measure change vs α=0 using PSNR/LPIPS.
        PSNR vs baseline should DECREASE as α increases (further from neutral).
        """
        if self.rank != 0:
            return None
        assert len(alphas) >= 2 and alphas[0] == 0.0, "alphas must start with 0.0"

        model  = self._model_unwrapped()
        device = self.device

        # ---- 1) Forward once to get condition windows and μ ----
        with self.amp_ctx():
            lam = model.forward(video_bnchw.float())

        cond_bnchw: Tensor = lam["condition_video_bnchw"]   # [B, N, C, H, W]
        mu_bn1d:    Tensor = lam["mean_bn1d"]               # [B, N, 1, D]

        # Flatten windows to M, keep N=1 convention for decode_to_frame
        cond_m1chw = eo.rearrange(cond_bnchw, "b n c h w -> (b n) 1 c h w")
        mu_m11d    = eo.rearrange(mu_bn1d,    "b n 1 d -> (b n) 1 1 d")

        M = cond_m1chw.shape[0]
        if max_eval_windows is not None and M > max_eval_windows:
            cond_m1chw = cond_m1chw[:max_eval_windows]
            mu_m11d    = mu_m11d[:max_eval_windows]
            M          = cond_m1chw.shape[0]

        A = len(alphas)

        # ---- 2) Decode predictions for all α in one batch ----
        cond_stack_bnchw = eo.repeat(cond_m1chw, "m n c h w -> (a m) n c h w", a=A)
        act_scaled_bn1d  = torch.cat([mu_m11d * alpha for alpha in alphas], dim=0)  # [(A*M),1,1,D]

        dec = model.decode_to_frame(cond_stack_bnchw.float(), act_scaled_bn1d.float())
        if isinstance(dec, dict):
            if   "reconstructed_video_bnchw" in dec: preds_bnchw = dec["reconstructed_video_bnchw"]
            elif "predicted_video_bnchw"     in dec: preds_bnchw = dec["predicted_video_bnchw"]
            elif "reconstructed_frame_nchw"  in dec: preds_bnchw = eo.rearrange(dec["reconstructed_frame_nchw"], "m c h w -> m 1 c h w")
            elif "frame_nchw"                in dec:
                out = dec["frame_nchw"]
                preds_bnchw = out if out.ndim == 5 else eo.rearrange(out, "m c h w -> m 1 c h w")
            else:
                raise KeyError(f"decode_to_frame() keys not recognised: {list(dec.keys())}")
        else:
            out = dec
            if out.ndim == 4: preds_bnchw = eo.rearrange(out, "m c h w -> m 1 c h w")
            elif out.ndim == 5: preds_bnchw = out
            else: raise RuntimeError(f"Unexpected decode shape: {tuple(out.shape)}")

        if self.loss_variant == "residual":
            preds_bnchw = preds_bnchw + cond_stack_bnchw

        preds_a_m_1_chw = eo.rearrange(preds_bnchw,      "(a m) n c h w -> a m n c h w", a=A, m=M)
        # cond_a_m_1_chw  = eo.rearrange(cond_stack_bnchw, "(a m) n c h w -> a m n c h w", a=A, m=M)  # for debugging

        # ---- 3) (If latent-space) decode to RGB for metrics; return NCHW ----
        def to_rgb_0_1(x_bnchw: Tensor) -> Tensor:
            if self.on_latents:
                x_nchw = eo.rearrange(x_bnchw, "b n c h w -> (b n) c h w").bfloat16()
                x_pix  = F.sigmoid(self.decoder(x_nchw)).float().clamp(0, 1)
                return x_pix
            else:
                return eo.rearrange(x_bnchw, "b n c h w -> (b n) c h w").float().clamp(0, 1)

        baseline_nchw = to_rgb_0_1(preds_a_m_1_chw[0])  # [M, C, H, W]

        # ---- 4) Metrics vs baseline ----
        if not hasattr(self, "_lpips"):
            self._lpips = piq.LPIPS(reduction="mean").to(device).eval()

        psnr_vs_alpha:  list[float] = []
        lpips_vs_alpha: list[float] = []

        for a_idx in range(1, A):
            p_nchw = to_rgb_0_1(preds_a_m_1_chw[a_idx])   # [M, C, H, W]
            b_nchw = baseline_nchw                        # [M, C, H, W]

            # Batch-mean PSNR (scalar)
            psnr_val = piq.psnr(p_nchw, b_nchw).item()
            psnr_vs_alpha.append(psnr_val)

            # Batch-mean LPIPS (scalar); expects [-1,1]
            lpips_val = self._lpips((p_nchw * 2 - 1), (b_nchw * 2 - 1)).item()
            lpips_vs_alpha.append(lpips_val)

        # ---- 5) Monotonicity score: PSNR should DECREASE as alpha increases ----
        def per_sample_psnr(x: Tensor, y: Tensor) -> Tensor:
            # returns [M]
            try:
                return piq.psnr(x, y, reduction="none")   # if supported by your piq version
            except TypeError:
                mse = F.mse_loss(x, y, reduction="none")  # [M,C,H,W]
                mse = mse.flatten(1).mean(dim=1)          # [M]
                return 10.0 * torch.log10(1.0 / mse.clamp_min(1e-10))

        per_window = []
        for a_idx in range(1, A):
            p_nchw = to_rgb_0_1(preds_a_m_1_chw[a_idx])   # [M, C, H, W]
            per_window.append(per_sample_psnr(p_nchw, baseline_nchw))  # [M]

        pw = torch.stack(per_window, dim=0) if per_window else torch.empty(0, M, device=device)
        if pw.numel() == 0:
            monotonicity_psnr = float("nan")
        else:
            # strictly decreasing across alphas
            diffs = pw[1:, :] - pw[:-1, :]    # [A-2, M]
            if diffs.numel() == 0:
                monotonicity_psnr = 1.0
            else:
                eps = 1e-6
                decr_mask = torch.all(diffs < -eps, dim=0)  # [M]
                monotonicity_psnr = decr_mask.float().mean().item()

        # ---- 6) Log (rank 0 only) ----
        if self._wandb_enabled:
            log_dict = {f"{log_key}/monotonicity_psnr": monotonicity_psnr}
            for a, v in zip(alphas[1:], psnr_vs_alpha):  log_dict[f"{log_key}/psnr_vs_alpha/alpha_{a:g}"]  = v
            for a, v in zip(alphas[1:], lpips_vs_alpha): log_dict[f"{log_key}/lpips_vs_alpha/alpha_{a:g}"] = v
            wandb.log({"global_step": self.global_step, **log_dict})

        return {
            "monotonicity_psnr": monotonicity_psnr,
            "psnr_vs_alpha":     dict(zip(alphas[1:], psnr_vs_alpha)),
            "lpips_vs_alpha":    dict(zip(alphas[1:], lpips_vs_alpha)),
        }


    @torch.no_grad()
    def evaluate(self) -> None:
        model: LatentActionModel = self._model_unwrapped()

        # ---- targets ----
        target_umap   = int(self.cfg.eval_num_samples_umap)
        target_recon  = int(self.cfg.eval_num_samples_recon)

        # ---- collectors ----
        latent_actions_list_bn1d: list[Tensor] = []
        recon_videos_list_bnchw: list[Tensor]  = []

        # ---- iterators (normal + long) ----
        ev_iter      = self.iter_dataset(sync_cuda_timing=False)
        ev_iter_long = self.iter_dataset(long_batch=True, sync_cuda_timing=False)
        vl_iter      = self.iter_dataset(validation=True)

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

            # -- action controllability --
            if self.rank == 0:
                _ = self.validate_action_controllability(recon_videos_bnchw, alphas=(0.0, 0.5, 1.0, 1.5, 2.0), max_eval_windows=64)

            # -- rollouts --
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
                    "global_step": self.global_step,
                    f"UMAP Scatter Plot/{self.global_step}":            scatter_plot,
                    f"Reconstruction/{self.global_step}":               video_table,
                    f"Rollout/Groundtruth    - {self.global_step}":     as_wandb_video(gt_rollout_nchw),
                    f"Rollout/Teacher.forced - {self.global_step}":     as_wandb_video(tf_rollout_nchw),
                    f"Rollout/Autoregressive - {self.global_step}":     as_wandb_video(ar_rollout_nchw),
                })

        print(f"[rank {self.rank}] validation done")
        barrier()


if __name__ == "__main__":
    config = LatentActionModelTrainingConfig.from_yaml("configs/lam_latent_debug.yml")
    trainer = Trainer_LatentActionModel(config)
    trainer.train()
