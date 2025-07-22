import  os
import  wandb
import  torch
import  einops as eo
from    datetime            import timedelta
from    typing              import Literal, Optional
from    torch               import Tensor
import  torch.distributed   as dist

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
            timeout=timedelta(minutes=30),
        )

    rank       = torch.distributed.get_rank()       if distributed else 0
    world_size = torch.distributed.get_world_size() if distributed else 1
    local_rank = int(os.environ.get("LOCAL_RANK", 0))

    if torch.cuda.is_available(): torch.cuda.set_device(local_rank) ; device = torch.device("cuda", local_rank)
    else: device = torch.device("cpu")

    return rank, world_size, device


def as_wandb_video(
        vid_nchw: torch.Tensor,
        title: str = "",
        fps: int = 4,
        fmt: Literal["gif", "mp4", "webm"] = "gif",
):
    if vid_nchw.dim() == 3: vid_nchw = vid_nchw.unsqueeze(0)
    vid_nchw = vid_nchw.detach().cpu()
    if vid_nchw.dtype.is_floating_point: vid_nchw = (vid_nchw.clamp(0, 1) * 255).to(torch.uint8)
    else:                                vid_nchw = vid_nchw.to(torch.uint8)
    vid_nchw_np = vid_nchw.numpy()
    return wandb.Video(vid_nchw_np, caption=title, fps=fps, format=fmt)


def get_world_size() -> int:
    world_size = torch.distributed.get_world_size() if torch.distributed.is_initialized() else 1
    return world_size


def barrier():
    if dist.is_initialized(): dist.barrier()


def broadcast_from_rank(tensor: torch.Tensor, rank: int = 0) -> torch.Tensor:
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.broadcast(tensor, src=rank)
    return tensor


def gather_to_rank(tensor:  Tensor,
                   dst:     int = 0,
                   dim:     int = 0,
                   cat:     bool = True) -> Optional[Tensor | list[Tensor]]:
    """
    Gather *unequal*-length tensors from all ranks to `dst`.

    Returns
    -------
    • On `dst`:  concatenated tensor  (if cat=True) or list of shards (if cat=False)
    • On others: None
    • If dist not initialised: the original `tensor`
    """
    if not (dist.is_available() and dist.is_initialized()): return tensor

    rank       = dist.get_rank()
    world_size = dist.get_world_size()

    # -- communicate local length
    local_len  = torch.tensor([tensor.shape[dim]], device=tensor.device)
    lens       = [torch.zeros_like(local_len) for _ in range(world_size)]
    dist.all_gather(lens, local_len)                         # every rank knows every length
    max_len    = int(torch.stack(lens).max())

    # -- pad so that all chunks have identical shape
    if tensor.shape[dim] < max_len:
        pad_shape      = list(tensor.shape)
        pad_shape[dim] = max_len - tensor.shape[dim]
        pad_chunk      = torch.zeros(*pad_shape,
                                     dtype=tensor.dtype,
                                     device=tensor.device)
        tensor         = torch.cat([tensor, pad_chunk], dim=dim)

    # -- gather to dst
    gather_list = [torch.empty_like(tensor) for _ in range(world_size)] if rank == dst else None
    dist.gather(tensor, gather_list=gather_list, dst=dst)

    if gather_list is None: return None

    # -- unpad + cat (on dst only)
    shards = []
    for chunk, ln in zip(gather_list, lens):
        slc         = [slice(None)]*chunk.dim()
        slc[dim]    = slice(0, int(ln))
        shards.append(chunk[tuple(slc)])

    return torch.cat(shards, dim=dim) if cat else shards
