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
import  toolz
import  einops as eo
import  plotly.express as px
from    torch.types import Number
from    torch       import Tensor
import  torch.nn.functional as F
import  plotly.figure_factory as ff
import  plotly.graph_objects as go
from    typing      import TypedDict, NotRequired

from latent_action_models.models.latent_action_model    import ActionEncodingInfo, LatentActionModel, LatentActionModelOutput
from latent_action_models.configs                       import LatentActionModelTrainingConfig
from latent_action_models.trainers.base_trainer         import BaseTrainer
from latent_action_models.action_creation.clustering    import umap_visualization
from latent_action_models.utils                         import (
    as_wandb_video, barrier, gather_to_rank, init_distributed, colors_labels_from_actions, get_quiver_vectors_from_actions, gather_objects_to_rank)
from latent_action_models.data_exploration              import create_actions_parquet as lam_parquet


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
    batch_sec:      NotRequired[float]
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
        conditioning    = config.conditioning
    )

def kl_divergence(mean: Tensor, logvar: Tensor) -> Tensor:
    return -0.5 * torch.sum(1 + logvar - mean ** 2 - logvar.exp(), dim=1).mean()

class Trainer_LatentActionModel(BaseTrainer):
    def __init__(self, config: LatentActionModelTrainingConfig) -> None:
        latent_action_model = create_latent_action_model(config)
        *_, device = init_distributed()
        super(Trainer_LatentActionModel, self).__init__(latent_action_model, config, device=device)
        self.beta               = 0.
        self._wandb_run         = None
        self.debug_show_samples = 10
        self.cfg: LatentActionModelTrainingConfig = config # -- reassign for typechecking:)

    @property
    def save_path(self) -> str: return f'lam_s{self.global_step}_beta{self.cfg.beta}.pt'

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

        return  (
            video_bnchw.to(self.device),
            metadata
        )


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
                                                            'b n c h w -> (b n) c h w').clip(0, 1)
                future_frame_video_nchw     = eo.rearrange(future_frame_video_bnchw,
                                                            'b n c h w -> (b n) c h w').clip(0, 1)
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
            if self.should_validate: 
                try: self.validate_simple()
                except Exception as e:
                    print(f'Validation failed')
                    import traceback
                    traceback.print_exc()
                    
            if bool(self.debug_show_samples) and self.should_log:
                wandb.log({
                    f'debug/sample_{self.debug_show_samples}_video': as_wandb_video((video_bnchw+1.)/2., "video"),
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
            latent_actions_bn1d = torch.cat(latent_actions_list_bn1d, dim=0)
            latent_actions_n1d  = eo.rearrange(latent_actions_bn1d, 'b n 1 d -> (b n) 1 d')
            latent_actions_n1d  = gather_to_rank(latent_actions_n1d, dst=0, dim=0, cat=True)

            recon_videos_bnchw = torch.cat(recon_videos_list_bnchw, dim=0)
            lam_outputs = model.forward(recon_videos_bnchw)
            condition_video_bnchw = gather_to_rank(lam_outputs["condition_video_bnchw"], dst=0, dim=0, cat=True)
            recon_video_bnchw = gather_to_rank(lam_outputs["reconstructed_video_bnchw"], dst=0, dim=0, cat=True)
            gt_video_bnchw = gather_to_rank(lam_outputs["groundtruth_video_bnchw"], dst=0, dim=0, cat=True)

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
                video_table = wandb.Table(columns=["conditioning", "predicted", "ground_truth"])
                for cond, recon, gt in zip(condition_video_bnchw, recon_video_bnchw, gt_video_bnchw):
                    video_table.add_data(as_wandb_video((cond+1.)/2.,  "Conditioning"),
                                        as_wandb_video((recon+1.)/2., "Predicted next-frame"),
                                        as_wandb_video((gt+1.)/2.,    "Ground-truth next-frame"))

                # --- Simplified logging ---
                if self._wandb_run:
                    print(f"[rank {self.rank}] logging to wandb")
                    wandb.log({
                        f"UMAP Scatter Plot/{self.global_step}": scatter_plot,
                        f"Reconstruction/{self.global_step}": video_table,
                    }, step=self.global_step)

        print(f"[rank {self.rank}] validation done")
        barrier()

    @torch.no_grad()
    def validate(self) -> None:
        # we assume we are going to have more umap samples than reconstruction samples (of course)
        model: LatentActionModel    = self._model_unwrapped()
        parquet: pl.DataFrame|None  = pl.read_parquet(lam_parquet.OUT_PARQUET) if self.cfg.data_config.dataset_name == 'gta_4' else None
        num_samples_umap            = self.cfg.val_num_samples_umap
        num_samples_recon           = self.cfg.val_num_samples_recon

        # -- umap visualization
        num_processed_umap_samples  = 0 ; _more_umap  = lambda: num_processed_umap_samples  < num_samples_umap
        num_processed_recon_samples = 0 ; _more_recon = lambda: num_processed_recon_samples < num_samples_recon
        latent_actions_list_bn1d    = []
        all_paths_local             = []
        all_idx_start_b_local       = []
        groundtruth_actions_list    = []
        recon_videos_list_bnchw     = []

        while _more_umap() or _more_recon():
            video_bnchw, paths, idx_start_b             = self  .format_batch()
            if _more_umap():
                action_info: ActionEncodingInfo         = model .encode_to_actions(video_bnchw)
                latent_actions_list_bn1d               += [action_info['mean_bn1d']]
                all_paths_local                        += paths
                all_idx_start_b_local                  += [idx_start_b]
                # fetches the actions for a certain video at a certain start index. does not handle strides yet
                num_umap_samples_in_batch               = action_info['mean_bn1d'].shape[0] * action_info['mean_bn1d'].shape[1]
                num_processed_umap_samples             += num_umap_samples_in_batch

            if _more_recon():
                recon_videos_list_bnchw                += [video_bnchw[:num_samples_recon, ::]]
                num_recon_samples_in_batch              = video_bnchw.shape[0]
                num_processed_recon_samples            += num_recon_samples_in_batch

        print(f"[rank {self.rank}] validation done with wandb_enabled={self._wandb_enabled}")

        if self._wandb_enabled:
            # -- umap 
            latent_actions_bn1d = torch.cat     (latent_actions_list_bn1d, dim=0)
            latent_actions_n1d  = eo.rearrange  (latent_actions_bn1d, 'b n 1 d -> (b n) 1 d')
            latent_actions_n1d  = gather_to_rank(latent_actions_n1d, dst=0, dim=0, cat=True)

            all_idx_start_b     = torch.cat(all_idx_start_b_local, dim=0)
            all_idx_start_b     = gather_to_rank(all_idx_start_b, dst=0, dim=0, cat=True)
            all_paths_global    = gather_objects_to_rank(all_paths_local, dst=0)
            # -- reconstruction of image
            recon_videos_bnchw                      = torch.cat(recon_videos_list_bnchw, dim=0)
            lam_outputs: LatentActionModelOutput    = model.forward(recon_videos_bnchw)
            condition_video_bnchw                   = gather_to_rank(lam_outputs["condition_video_bnchw"],     dst=0, dim=0, cat=True)
            recon_video_bnchw                       = gather_to_rank(lam_outputs["reconstructed_video_bnchw"], dst=0, dim=0, cat=True)
            gt_video_bnchw                          = gather_to_rank(lam_outputs["groundtruth_video_bnchw"],   dst=0, dim=0, cat=True)

            if self.rank == 0:
                # -- umap
                groundtruth_actions_list               += [
                    parquet.filter( (pl.col('video_path')   == p)  &
                                    (pl.col('frame_idx')    >= i.item()) &
                                    (pl.col('frame_idx')    <  i.item() + action_info['mean_bn1d'].shape[1]))
                    if      parquet is not None else 'NO_ACTION'
                    for     p,i in zip(all_paths_global, all_idx_start_b)
                ]
                colors, labels, legend          = colors_labels_from_actions(groundtruth_actions_list, parquet, num_frames_per_clip=action_info['mean_bn1d'].shape[1])
                u, v                            = get_quiver_vectors_from_actions(groundtruth_actions_list, num_frames_per_clip=action_info['mean_bn1d'].shape[1])
                umap_embed_n2, fig              = umap_visualization(latent_actions_n1d,
                                                                     colors=colors,
                                                                     legend=legend,
                                                                     vis_filename=f'umap_visualization_{self.save_path}'.replace('.pt', ''))

                umap_df = pd.DataFrame({
                    'umap_x': umap_embed_n2[:, 0],
                    'umap_y': umap_embed_n2[:, 1],
                    'action_label': labels, 'colors': colors,
                    'u': u, 'v': v
                })
                fig = go.Figure()
                unique_labels   = umap_df['action_label'].unique()
                color_map       = dict(zip(unique_labels, px.colors.qualitative.Set1[:len(unique_labels)]))
                for label in unique_labels:
                    label_data = umap_df[umap_df['action_label'] == label]

                    quiver_fig = ff.create_quiver(
                        label_data['umap_x'], label_data['umap_y'], label_data['u'], label_data['v'],
                        scale=0.4, arrow_scale=0.25, line=dict(color=color_map[label], width=2)
                    )

                    for trace in quiver_fig.data:
                        trace.name = f'Arrows - {label}'
                        trace.showlegend = False
                        fig.add_trace(trace)
                    
                    fig.add_trace(go.Scatter(
                        x=label_data['umap_x'], y=label_data['umap_y'],
                        mode='markers', marker=dict(size=7, color=color_map[label], opacity=0.75),
                        name=f'Points - {label}', text=[f'u: {u_val:.3f}, v: {v_val:.3f}' for u_val, v_val in zip(label_data['u'], label_data['v'])],
                        hovertemplate=f'<b>{label}</b><br>%{{text}}<br>X: %{{x:.3f}}<br>Y: %{{y:.3f}}<extra></extra>'
                    ))
                
                fig.update_layout(
                    title='UMAP Visualization with Colored Arrows',
                    xaxis_title='UMAP Component 1',
                    yaxis_title='UMAP Component 2'
                )


                vis_path = f"visualizations/umap_vis_colored_arrows_{self.global_step}.html"
                fig.write_html(vis_path, auto_play=False)
                scatter_table = wandb.Table(columns=['scatter_plot'])
                scatter_table.add_data(wandb.Html(vis_path))
                # -- reconstruction
                video_table = wandb.Table(columns=["conditioning", "predicted", "ground_truth"])
                for cond, recon, gt in zip(condition_video_bnchw, recon_video_bnchw, gt_video_bnchw):
                    video_table.add_data(as_wandb_video(cond,  "Conditioning"),
                                         as_wandb_video(recon, "Predicted next-frame"),
                                         as_wandb_video(gt,    "Ground-truth next-frame"))

                if self._wandb_run:
                    print(f"[rank {self.rank}] logging to wandb")
                    wandb.log({
                        f"UMAP Table (Interactive)":     scatter_table,
                        f"Reconstruction":               video_table,
                        f"UMAP Plot (Static)":           fig,
                    },  step=self.global_step)

        print(f"[rank {self.rank}] validation done")
        barrier()


if __name__ == "__main__":
    config = LatentActionModelTrainingConfig.from_yaml("configs/lam.yml")
    trainer = Trainer_LatentActionModel(config)
    trainer.train()
