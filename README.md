# Image-to-Text Retrieval

Train a cross-modal retrieval model that searches photos using natural language.  
EfficientNet-B0 (image) + DistilBERT (text) trained with InfoNCE contrastive learning on MS COCO 2017.

**CNIT 58100AMC · Purdue University · Spring 2026**  
Johan Prince · Gavin Oxley · Aryan Singh · Selina Shim

---

## Project Structure

```
image-to-text/
├── dataset.py              # COCO data loading, 224×224 resize, DistilBERT tokenization
├── model.py                # EfficientNet-B0 image encoder + DistilBERT text encoder (256-dim)
├── train.py                # InfoNCE contrastive training loop
├── evaluate.py             # Recall@K evaluation on validation set
├── export.py               # ONNX export + Qualcomm AI Hub INT8 compile job
├── inference.py            # On-device inference via Qualcomm AI Hub
├── requirements.txt        # Local / Qualcomm deployment (torch 2.2.2 pinned)
├── requirements-train.txt  # SageMaker / cloud GPU (torch 2.4+, NumPy 2.x compatible)
├── train2017/              # COCO training images (118K)
├── val2017/                # COCO validation images (5K)
├── test2017/               # COCO test images
├── annotations/            # COCO caption annotation JSON files
└── checkpoints/            # Saved model checkpoints
```

---

## 1. Prerequisites

- Python 3.11
- [Kaggle API token](https://www.kaggle.com/settings/account) (`~/.kaggle/kaggle.json`)
- [Qualcomm AI Hub API token](https://aihub.qualcomm.com) (required for `export.py` and `inference.py`)

---

## 2. Download MS COCO 2017 Dataset

```bash
# Download the full COCO 2017 dataset from Kaggle (~26 GB)
curl -L -o coco-2017-dataset.zip \
  https://www.kaggle.com/api/v1/datasets/download/awsaf49/coco-2017-dataset

# Extract into the project directory
unzip coco-2017-dataset.zip -d /path/to/image-to-text
```

After extraction, the directory structure should look like this:

```
image-to-text/
├── train2017/          # 118,287 images
├── val2017/            # 5,000 images
├── test2017/           # 40,670 images
└── annotations/
    ├── captions_train2017.json
    └── captions_val2017.json
```

---

## 3. Environment Setup

### Local (macOS / Qualcomm deployment)

```bash
# Create a Python 3.11 virtual environment
python3.11 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

> **Note:** `transformers >= 5.0` requires `torch >= 2.4`. Since `qai-hub-models` pins `torch==2.2.2`,
> `transformers` is pinned to `< 5.0` in `requirements.txt`.

### SageMaker / Cloud GPU

```bash
# Use the training-only requirements (torch 2.4+, NumPy 2.x compatible)
pip install -r requirements-train.txt
```

> **Note:** `torchvision==0.17.2` was compiled against NumPy 1.x and crashes with NumPy 2.x.
> `requirements-train.txt` uses `torch>=2.4` + `torchvision>=0.19` which support NumPy 2.x natively.

### CUDA environment (Linux / Windows)

```bash
pip install torch==2.2.2 torchvision==0.17.2 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt --ignore-installed torch torchvision
```

---

## 4. Training

```bash
# Quick sanity check (1,000 samples, 1 epoch — finishes in a few minutes)
python train.py --epochs 1 --max_samples 1000 --freeze_epochs 0

# Mid-scale test (10,000 samples, 3 epochs)
python train.py --epochs 3 --max_samples 10000

# Full training (118K samples, 10 epochs — ~6 hours on Apple M3 Pro)
python train.py --epochs 10
```

| Option | Default | Description |
|--------|---------|-------------|
| `--epochs` | 10 | Number of training epochs |
| `--batch_size` | 64 | Batch size |
| `--lr` | 1e-4 | Learning rate |
| `--embed_dim` | 256 | Embedding dimension |
| `--freeze_epochs` | 2 | Epochs to keep backbones frozen |
| `--max_samples` | None (full dataset) | Limit training to the first N samples |
| `--save_dir` | checkpoints | Directory to save checkpoints |

Checkpoints are saved as `checkpoints/epoch{N}.pt` after each epoch, and as `checkpoints/final_model.pt` when training completes.

### Background training (keep running after closing terminal)

```bash
nohup python train.py --epochs 10 > train.log 2>&1 &
echo $! > train.pid

# Monitor progress
tail -f train.log

# Stop training
kill $(cat train.pid)
```

---

## 5. Evaluation

```bash
# Evaluate the final model (default: checkpoints/final_model.pt)
python evaluate.py

# Evaluate a specific checkpoint
python evaluate.py --checkpoint checkpoints/epoch03.pt
```

| Option | Default | Description |
|--------|---------|-------------|
| `--checkpoint` | checkpoints/final_model.pt | Path to the checkpoint to evaluate |
| `--embed_dim` | 256 | Embedding dimension (must match training) |

**Target performance on COCO 2017 val (5K):**

| Direction | R@1 | R@5 | R@10 | Median Rank |
|-----------|-----|-----|------|-------------|
| Image → Text | 38.2% | 65.1% | 74.8% | 7.3 |
| Text → Image | 29.4% | 57.3% | 68.2% | 11.3 |

---

## 6. Export & Mobile Deployment

```bash
# Configure Qualcomm AI Hub (first time only)
qai-hub configure --api_token <YOUR_TOKEN>

# List available target devices
python export.py --list_devices

# Export to ONNX and submit INT8 compile job
python export.py --device "Samsung Galaxy S24"
```

| Option | Default | Description |
|--------|---------|-------------|
| `--checkpoint` | checkpoints/final_model.pt | Model checkpoint to export |
| `--embed_dim` | 256 | Embedding dimension |
| `--device` | Samsung Galaxy S24 | AI Hub target device name |
| `--list_devices` | — | Print all available devices and exit |

`export.py` performs the following steps:
1. Exports `image_encoder.onnx` (static shape `[1, 3, 224, 224]`) and `text_encoder.onnx` (static shape `[1, 64]`)
2. Submits INT8 quantization compile jobs to Qualcomm AI Hub
3. Compile job IDs are printed to the terminal and visible in [AI Hub Workbench](https://app.aihub.qualcomm.com)

> **Known issues:**
> - Use static shapes only — dynamic shapes are not supported by Qualcomm AI Hub
> - `int64` inputs require `--truncate_64bit_io` compile option (already included)

---

## 7. On-Device Inference

After export, use the compile job IDs printed by `export.py`:

```bash
# Run inference with a query image and text
python inference.py \
  --image_job <image_compile_job_id> \
  --text_job  <text_compile_job_id> \
  --image val2017/000000481404.jpg \
  --query "a dog running on the beach"

# Profile job — detailed per-layer latency breakdown
python inference.py \
  --image_job <image_compile_job_id> \
  --text_job  <text_compile_job_id> \
  --profile
```

| Option | Default | Description |
|--------|---------|-------------|
| `--image_job` | (required) | Compile job ID for the image encoder |
| `--text_job` | (required) | Compile job ID for the text encoder |
| `--image` | images/dog.jpg | Path to query image |
| `--query` | "a photo of a dog" | Natural language query text |
| `--device` | Samsung Galaxy S24 | AI Hub target device name |
| `--profile` | — | Submit profile jobs for latency analysis |

**On-device performance (Snapdragon, QNN HTP):**

| Component | ONNX size | INT8 size | Latency |
|-----------|-----------|-----------|---------|
| Image encoder | 47 MB | 12 MB | 14.2 ms |
| Text encoder | 63 MB | 16 MB | 11.8 ms |
| Combined | 110 MB | 28 MB | ~26 ms/query |

---

## 8. Model Architecture

```
Image ──► EfficientNet-B0 (pretrained) ──► Linear(1280→256) ──► LayerNorm ──► L2 norm ──►┐
                                                                                           ├──► cosine similarity
Text  ──► DistilBERT [CLS] (pretrained) ──► Linear(768→256)  ──► LayerNorm ──► L2 norm ──►┘
```

- **Loss**: Symmetric InfoNCE with learnable temperature parameter (τ initialized to 0.07)
- **Dataset**: MS COCO 2017 — 118K training / 5K validation image-caption pairs
