export PYTHONPATH=owl-data-2/:$PYTHONPATH
echo "Creating virtual environment..."
uv venv venv --python 3.12
source venv/bin/activate

echo "Installing dependencies..."
uv pip install -e .

echo "Installing ffmpeg..."
sudo apt install -y ffmpeg


echo "Launching video loader server..."
python -m owl_data.video_loader_server \
    --root_dir /mnt/data/sami/1x_dataset/original/train_v2.0_raw/videos \
    --num_workers 64 \
    --queue_max 200000 \
    --frame_skip 1 \
    --n_frames 2 \
    --known_fps 30 \
    --suffix .mp4

echo "Soft linking checkpoints..."
ln -s /mnt/data/sami/checkpoints/lam checkpoints_sl/

echo "Logging into wandb..."

wandb login