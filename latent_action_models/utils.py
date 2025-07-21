import  os
import  wandb
import  torch
import  einops as eo
from    torch import Tensor

# reused from adaworld repo 

def patchify(videos_bnchw: Tensor, size: int) -> Tensor:
    *_, H, W     = videos_bnchw.shape
    videos_bnchw    = videos_bnchw[:,:, :, 
                                    :H - (H % size), # -- round down to nearest patch size
                                    :W - (W % size)]
    patches_bnpd    = eo.rearrange(videos_bnchw,
                                "b t c (hn hp) (wn wp) -> b t (hn wn) (hp wp c)",
                                hp=size, wp=size)
    return patches_bnpd


def unpatchify(patches_bnpd: Tensor, size: int, h_out: int, w_out: int) -> Tensor:
    h_pad       = -h_out % size
    hn          = (h_out + h_pad) // size
    video_bnhwc = eo.rearrange(patches_bnpd,
                            "b t (hn wn) (hp wp c) -> b t c (hn hp) (wn wp)",
                            hp=size, wp=size, hn=hn)
    return video_bnhwc[:,:,:h_out, :w_out]
 

def init_distributed() -> tuple[int, int, torch.device]:
    distributed = int(os.environ.get("WORLD_SIZE", "1")) > 1

    if distributed and not torch.distributed.is_initialized():
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        torch.distributed.init_process_group(
            backend=backend,
            timeout=torch.timedelta(minutes=30),
        )

    rank       = torch.distributed.get_rank()       if distributed else 0
    world_size = torch.distributed.get_world_size() if distributed else 1
    local_rank = int(os.environ.get("LOCAL_RANK", 0))

    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
    else:
        device = torch.device("cpu")

    return rank, world_size, device

def as_wandb_video(video_nchw: Tensor, title: str) -> wandb.Video:
    pass