# NVDEC setup for DALI GPU video decoding

DALI’s GPU/*mixed* video decoders (`readers.video`, `experimental.decoders.video`) look for the **NVDEC runtime stub** `libnvcuvid.so` at import‑time.  On many minimal Ubuntu clusters that library isn’t installed by default.  Below is one way to make it visible *without* Docker.

## User‑space (no root)

```bash
# 1 download the driver that matches `nvidia-smi`
wget https://us.download.nvidia.com/tesla/550.163.01/NVIDIA-Linux-x86_64-550.163.01.run
sh NVIDIA-Linux-x86_64-550.163.01.run --extract-only --target ~/nvidia_driver

# 2 copy NVDEC libs into a private dir
mkdir -p ~/nvidia_libs
cp ~/nvidia_driver/libnvcuvid.so.550.163.01  ~/nvidia_libs/
ln -s libnvcuvid.so.550.163.01 ~/nvidia_libs/libnvcuvid.so.1
ln -s libnvcuvid.so.550.163.01 ~/nvidia_libs/libnvcuvid.so

# 3 expose to the loader
echo 'export LD_LIBRARY_PATH=$HOME/nvidia_libs:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc
```

Verify:

```python
import ctypes
ctypes.CDLL("libnvcuvid.so")  # no OSError ⇒ ready
```

Once `libnvcuvid.so` resolves, you can switch your DALI pipeline back to `device="mixed"` or `device="gpu"`.
Note that `device="cpu"` would not work for video reading.
---

## Tip

If you share the installation directory across nodes (e.g. under `/opt/nvidia_libs`) just add that path to `LD_LIBRARY_PATH` in your environment module or cluster profile.
