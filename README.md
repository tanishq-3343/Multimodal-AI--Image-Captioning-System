# 🖼️ Image Captioning with Vision + Transformer

> Automatically generate text descriptions from images using ResNet-50 + Transformer Decoder.

---

## 📋 Table of Contents
- [Overview](#overview)
- [Project Structure](#project-structure)
- [Requirements](#requirements)
- [Setup](#setup)
- [Dataset](#dataset)
- [Usage](#usage)
- [Configuration](#configuration)
- [Expected Results](#expected-results)
- [Troubleshooting](#troubleshooting)

---

## Overview

This project implements an end-to-end image captioning system:

```
Input Image → ResNet-50 Encoder → Transformer Decoder → Caption
```

**Example:**
```
Input:  [Image of a dog in a park]
Output: "A brown dog playing with a ball in the park"
```

**Target Performance:**
| Metric | Target |
|--------|--------|
| BLEU-4 | ≥ 0.25 |
| CIDEr  | ≥ 1.0  |
| METEOR | ≥ 0.25 |

---

## Project Structure

```
image_captioning/
├── dataset.py          # Data loading and preprocessing
├── models.py           # Vision Encoder + Transformer Decoder
├── train.py            # Training loop
├── inference.py        # Caption generation
├── evaluate.py         # BLEU score calculation
├── main.py             # Main execution script
├── requirements.txt    # Python dependencies
├── README.md           # This file
├── checkpoints/        # Saved model weights (auto-created)
└── flickr8k/           # Dataset directory (you download this)
    ├── Images/
    └── captions.txt
```

---

## Requirements

- Python 3.8+
- PyTorch 2.0+
- 8GB+ RAM
- GPU/MPS recommended (CPU works but slow)

---

## Setup

### 1. Clone or create project directory
```bash
mkdir image_captioning
cd image_captioning
```

### 2. Create virtual environment
```bash
# Mac/Linux
python3 -m venv venv
source venv/bin/activate

# Windows
python -m venv venv
venv\Scripts\activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

---

## Dataset

### Download Flickr8k (Recommended)
1. Go to: https://www.kaggle.com/datasets/adityajn105/flickr8k
2. Download and extract
3. Place in project directory:

```
flickr8k/
├── Images/         ← ~8,000 images
└── captions.txt    ← 40,000 captions
```

### Dataset Stats
| Property | Value |
|----------|-------|
| Total Images | 8,000 |
| Captions per Image | 5 |
| Total Captions | 40,000 |
| Avg Caption Length | 10-12 words |
| Disk Size | ~1 GB |

---

## Usage

### Run Full Pipeline
```bash
python main.py
```

### Step-by-Step

**1. Test data loading only**
```python
from dataset import get_dataloader

loader, dataset = get_dataloader(
    root_dir='./flickr8k/Images',
    captions_file='./flickr8k/captions.txt',
    batch_size=32
)
print(f"Vocab size: {len(dataset.vocab)}")
print(f"Dataset size: {len(dataset)}")
```

**2. Generate caption for a single image**
```python
import torch
from models import ImageCaptioningModel
from inference import generate_caption

device = torch.device("mps")  # or "cuda" or "cpu"

# Load trained model
checkpoint = torch.load('./checkpoints/best_model.pth')
model = ImageCaptioningModel(vocab_size=8254).to(device)
model.load_state_dict(checkpoint['model_state_dict'])

# Generate caption
caption = generate_caption(
    model=model,
    image_path='./flickr8k/Images/your_image.jpg',
    vocab=dataset.vocab,
    device=device
)
print(f"Caption: {caption}")
```

**3. Evaluate BLEU score**
```python
from evaluate import evaluate_bleu

bleu = evaluate_bleu(model, loader, dataset.vocab, device, num_samples=500)
print(f"BLEU-4: {bleu:.4f}")
```

---

## Configuration

Edit `main.py` to change settings:

```python
config = {
    'data_dir':       './flickr8k/Images',  # Path to images
    'captions_file':  './flickr8k/captions.txt',  # Path to captions
    'batch_size':     32,     # Reduce to 16 if out of memory
    'embed_size':     512,    # 512 (beginner) or 768 (advanced)
    'num_layers':     6,      # Transformer decoder layers
    'num_heads':      8,      # Attention heads
    'learning_rate':  1e-4,   # Adam optimizer LR
    'num_epochs':     20,     # Training epochs (20-50)
}
```

### Beginner Setup (~65M params)
```python
'embed_size': 512,
'num_layers': 6,
'num_heads':  8,
# Expected BLEU-4: 0.22-0.25
```

### Advanced Setup (~150M params)
```python
'embed_size': 768,
'num_layers': 8,
'num_heads':  12,
# Expected BLEU-4: 0.26-0.30
```

---

## Expected Results

### Training Progress
```
Using MPS (Apple Silicon GPU)

=== Loading Dataset ===
Vocabulary size: 8254
Number of training samples: 40455

=== Initializing Model ===
Total trainable parameters: 40,132,608

=== Starting Training ===
Epoch 1/20:  loss: 3.421
Epoch 5/20:  loss: 2.108
Epoch 10/20: loss: 1.534
Epoch 15/20: loss: 1.021
Epoch 20/20: loss: 0.812

=== Evaluating Model ===
BLEU-4 Score: 0.261

=== Training Complete ===
Final BLEU-4 Score: 0.2610
Target: ≥ 0.25
Status: ✓ ACHIEVED
```

### Training Time
| Hardware | Time per Epoch | Total (20 epochs) |
|----------|---------------|-------------------|
| MPS (Apple Silicon) | ~10-15 min | ~3-5 hours |
| NVIDIA GPU (RTX 3060) | ~5-6 min | ~2 hours |
| CPU only | ~60-90 min | ~24 hours |

---

## Troubleshooting

### Out of Memory
```python
# Reduce batch size in main.py
'batch_size': 16  # or 8
```

### MPS Errors (Apple Silicon)
```bash
export PYTORCH_ENABLE_MPS_FALLBACK=1
python main.py
```

### Slow Training on CPU
```bash
# Force MPS if on Mac
python -c "import torch; print(torch.backends.mps.is_available())"
# Should print: True
```

### NLTK Error
```python
import nltk
nltk.download('punkt')
nltk.download('punkt_tab')
```

### Import Errors
```bash
# Verify all files exist
ls *.py
# Should show: dataset.py models.py train.py inference.py evaluate.py main.py
```

---

## How It Works

```
1. IMAGE (224×224)
        ↓
2. ResNet-50 Encoder
   → Extracts 49 spatial features (7×7 grid)
   → Each feature = 512 dimensions
        ↓
3. Transformer Decoder (6 layers)
   → Self-Attention:  words attend to each other
   → Cross-Attention: words attend to image regions
   → Feed-Forward:    transforms representations
        ↓
4. Linear + Softmax
   → Predicts probability of each word
        ↓
5. OUTPUT: "A dog playing in the park"
```

---


## References
- [Flickr8k Dataset](https://www.kaggle.com/datasets/adityajn105/flickr8k)
- [PyTorch Documentation](https://pytorch.org/docs/)

---
