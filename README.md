# Image-to-Text Retrieval

Train a cross-modal retrieval model that searches photos using natural language.
EfficientNet-B0 (image) + DistilBERT (text) trained with InfoNCE contrastive learning on MS COCO 2017.

**CNIT 58100AMC · Purdue University · Spring 2026**
Johan Prince · Gavin Oxley · Aryan Singh · Selina Shim

---

## Project Structure

```
image-to-text/
├── dataset.py       # COCO data loading, 224×224 resize, DistilBERT tokenization
├── model.py         # EfficientNet-B0 image encoder + DistilBERT text encoder (256-dim)
├── train.py         # InfoNCE contrastive training loop
├── evaluate.py      # Recall@K evaluation on validation set
├── export.py        # ONNX export + Qualcomm AI Hub INT8 compile job
├── requirements.txt
├── train2017/       # COCO training images (118K)
├── val2017/         # COCO validation images (5K)
├── test2017/        # COCO test images
├── annotations/     # COCO caption annotation JSON files
└── checkpoints/     # Saved model checkpoints
```

---

## 1. Prerequisites

- Python 3.11
- [Kaggle API token](https://www.kaggle.com/settings/account) (`~/.kaggle/kaggle.json`)
- [Qualcomm AI Hub API token](https://aihub.qualcomm.com) (required for `export.py` only)

---

## 2. Download MS COCO 2017 Dataset

```bash
# Download the full COCO 2017 dataset from Kaggle (~26 GB)
curl -L -o ./coco-2017-dataset.zip \
  https://www.kaggle.com/api/v1/datasets/download/awsaf49/coco-2017-dataset

# Extract into the project directory
unzip ./coco-2017-dataset.zip -d /path/to/image-to-text
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

```bash
# Create a Python 3.11 virtual environment
python3.11 -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

> **CUDA environment (Linux / Windows):**
> ```bash
> pip install torch==2.2.2 torchvision==0.17.2 --index-url https://download.pytorch.org/whl/cu121
> pip install -r requirements.txt --ignore-installed torch torchvision
> ```

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

# Export to ONNX and submit INT8 compile job to AI Hub
python export.py
```

`export.py` performs the following steps:
1. Exports `image_encoder.onnx` (47 MB) and `text_encoder.onnx` (63 MB)
2. Submits INT8 quantization compile jobs to Qualcomm AI Hub targeting the Snapdragon XR2 Gen 2
3. On-device performance after compilation: image encoder **14.2 ms**, text encoder **11.8 ms**

---

## 7. Model Architecture

```
Image ──► EfficientNet-B0 (pretrained) ──► Linear(1280→256) ──► LayerNorm ──► L2 norm ──►┐
                                                                                           ├──► cosine similarity
Text  ──► DistilBERT [CLS] (pretrained) ──► Linear(768→256)  ──► LayerNorm ──► L2 norm ──►┘
```

- **Loss**: Symmetric InfoNCE with learnable temperature parameter (τ initialized to 0.07)
- **Dataset**: MS COCO 2017 — 118K training / 5K validation image-caption pairs
