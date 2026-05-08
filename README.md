# 📄 VLM QLoRA Fine-Tuning — Document to Markdown Generation

Fine-tuning **Qwen2-VL-2B-Instruct** with **QLoRA** to convert document page images into structured Markdown text. Runs entirely on Kaggle's free dual T4 GPU environment.

![Python](https://img.shields.io/badge/Python-3.10-blue?style=flat-square)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-orange?style=flat-square)
![HuggingFace](https://img.shields.io/badge/HuggingFace-Transformers-yellow?style=flat-square)
![PEFT](https://img.shields.io/badge/PEFT-QLoRA-green?style=flat-square)
![Platform](https://img.shields.io/badge/Platform-Kaggle_T4×2-lightgrey?style=flat-square)

---

## Overview

Given a document image (academic paper, report, textbook page), the model generates the corresponding Markdown — preserving headings, lists, tables, equations, and document structure.

The model is trained using **QLoRA**: the base model is loaded in 4-bit quantized form and kept frozen, while lightweight LoRA adapter layers are trained on top. This makes fine-tuning a 2B parameter VLM feasible on consumer-grade GPUs.

---

## Demo

Upload a document image → get structured Markdown output instantly via the included Gradio app.

```
Input:  [scanned document page image]

Output: # Section Title
        Some paragraph text here...
        - bullet point one
        - bullet point two
        | Column A | Column B |
        |----------|----------|
        | value    | value    |
```

---

## Model & Dataset

| Component | Detail |
|-----------|--------|
| Base Model | [Qwen2-VL-2B-Instruct](https://huggingface.co/Qwen/Qwen2-VL-2B-Instruct) |
| Fine-Tuning | QLoRA (4-bit NF4 + LoRA adapters) |
| Dataset | [Nougat Training Dataset Example](https://www.kaggle.com/datasets/zphilip/nougat-training-dataset-example) |
| Platform | Kaggle T4 × 2 |
| Framework | PyTorch + HuggingFace Transformers + PEFT |

---

## Project Structure

```
├── vlm_qlora_finetune.ipynb   # Main Kaggle notebook (all parts)
├── README.md
└── outputs/
    ├── loss_curve.png          # Training & validation loss plot
    ├── val_pred_*.png          # Validation predictions (image | GT | generated)
    ├── train_pred_*.png        # Training image predictions
    ├── unseen_pred_*.png       # Unseen image predictions
    ├── compare_zs_ft_*.png     # Zero-shot vs fine-tuned comparison
    ├── rouge_metrics.csv       # ROUGE-1/2/L scores

```

---

## Quickstart

### 1. Run on Kaggle (recommended)

1. Go to [Kaggle](https://www.kaggle.com) and create a new notebook
2. Set accelerator to **GPU T4 × 2**
3. Add the [Nougat dataset](https://www.kaggle.com/datasets/zphilip/nougat-training-dataset-example) as input
4. Upload `vlm_qlora_finetune.ipynb` and run all cells

### 2. Reload after kernel restart

If your Kaggle session expires and the model is already saved, use this reload snippet:

```python
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor, BitsAndBytesConfig
from peft import PeftModel
import torch

ADAPTER_DIR = '/kaggle/input/models/<your-username>/qlora-finetuning/pytorch/default/1'

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type='nf4',
    bnb_4bit_compute_dtype=torch.float16,
)

base_model = Qwen2VLForConditionalGeneration.from_pretrained(
    'Qwen/Qwen2-VL-2B-Instruct',
    quantization_config=bnb_config,
    device_map='auto',
    torch_dtype=torch.float16,
    trust_remote_code=True,
)

inf_model = PeftModel.from_pretrained(base_model, ADAPTER_DIR, is_trainable=False)
inf_model.eval()

inf_processor = AutoProcessor.from_pretrained(
    ADAPTER_DIR, trust_remote_code=True,
    max_pixels=512*512, min_pixels=256*256,
)
```

---

## Training Configuration

| Hyperparameter | Value |
|----------------|-------|
| Epochs | 3 |
| Batch size | 1 |
| Gradient accumulation | 8 steps |
| Effective batch size | 8 |
| Learning rate | 1.5e-4 (cosine decay) |
| LoRA rank | 16 |
| LoRA alpha | 32 |
| LoRA dropout | 0.05 |
| Quantization | 4-bit NF4 |
| Image size | 512px (longest edge) |
| Max sequence length | 1024 tokens |
| Optimizer | Paged AdamW 8-bit |

---

## Notebook Sections

| Part | Description |
|------|-------------|
| Part 1 | Dataset exploration — image/markdown pair discovery and visualization |
| Part 2 | Data preparation — ChatML format conversion |
| Part 3 | Dataset splitting — 80% train / 20% validation |
| Part 4 | QLoRA fine-tuning — model loading, LoRA setup, training loop |
| Part 5 | Markdown generation — inference on validation set + ROUGE evaluation |
| Part 6 | Testing — 3 training images + 3 unseen document images |
| Bonus | Zero-shot vs fine-tuned comparison |
| App | Gradio demo with rendered Markdown preview |

---

## Kernel Restart Recovery

Kaggle sessions expire. This notebook handles it automatically:

- Checkpoints saved every epoch to `/kaggle/working/qlora_checkpoints/`
- A lightweight `training_log.json` records the last completed epoch and step
- On restart, `find_latest_checkpoint()` detects the most recent checkpoint and resumes via `trainer.train(resume_from_checkpoint=...)`
- Only the last 3 checkpoints are kept on disk to manage storage

---

## Requirements

```
transformers>=4.45.0
peft>=0.12.0
bitsandbytes>=0.43.0
accelerate>=0.34.0
trl>=0.11.0
qwen-vl-utils
torch>=2.0
torchvision
datasets
gradio
Pillow
rouge-score
matplotlib
pandas
```

All packages are installed automatically in Cell 1 of the notebook.

---

## Acknowledgements

- [Qwen Team](https://huggingface.co/Qwen) for Qwen2-VL-2B-Instruct
- [Meta Nougat](https://github.com/facebookresearch/nougat) for the dataset format
- [HuggingFace PEFT](https://github.com/huggingface/peft) for the LoRA implementation
- [Tim Dettmers et al.](https://arxiv.org/abs/2305.14314) for the QLoRA paper

---

## License

MIT
