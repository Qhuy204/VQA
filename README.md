# 🇻🇳 Qwen3-VL Finetuning on VietTravelVQA

Fine-tune **Qwen3-VL** (Vision-Language Model) trên bộ dữ liệu **VietTravelVQA** - Dataset VQA về du lịch Việt Nam.

Sử dụng **[Unsloth](https://unsloth.ai)** để tối ưu training với **4-bit quantization** và **LoRA**.

## ✨ Features

- 🦥 **Unsloth Optimization** - Training nhanh hơn 2x, tiết kiệm VRAM
- 📦 **4-bit Quantization** - Chạy được trên GPU 8GB+
- 🔧 **LoRA Finetuning** - Chỉ train adapter, không cần full model
- 🌐 **HuggingFace Integration** - Load data từ `Qhuy204/VQA`
- 🖼️ **Vision-Language** - Hỗ trợ câu hỏi về hình ảnh

## 🚀 Quick Start

### Installation

```bash
# Clone repo
git clone https://github.com/Qhuy204/VQA.git
cd VQA

# Install dependencies
pip install unsloth
pip install -r requirements.txt
```

### Training

```bash
# Load data từ HuggingFace (Qhuy204/VQA)
python finetune_qwen3vl.py --hf_dataset Qhuy204/VQA

# Hoặc từ local files
python finetune_qwen3vl.py \
    --train_file ./VietTravelVQA/viettravelvqa_train.json \
    --image_dir ./VietTravelVQA/images

# Test nhanh với 50 samples
python finetune_qwen3vl.py --hf_dataset Qhuy204/VQA --max_samples 50 --max_steps 30
```

### Inference

```bash
python inference.py \
    --model_path outputs/qwen3vl-viettravelvqa/lora_model \
    --image path/to/image.jpg \
    --question "Đây là địa điểm du lịch nào?"
```

## 📊 Dataset

**VietTravelVQA** - Visual Question Answering về du lịch Việt Nam:

| Split | Images | QA Pairs |
|-------|--------|----------|
| Train | 1,124 | 5,620 |
| Test | 282 | 1,410 |

- 🏛️ Di tích lịch sử, văn hóa
- 🏖️ Địa điểm du lịch
- 🍜 Ẩm thực địa phương
- 🎭 Lễ hội truyền thống

**Data Source:** [Qhuy204/VQA](https://huggingface.co/datasets/Qhuy204/VQA)

## ⚙️ Configuration

### Model Options

| Model | VRAM | HuggingFace |
|-------|------|-------------|
| 2B | ~6GB | `unsloth/Qwen3-VL-2B-Instruct-unsloth-bnb-4bit` |
| **4B** | ~8GB | `unsloth/Qwen3-VL-4B-Instruct-unsloth-bnb-4bit` |
| 8B | ~12GB | `unsloth/Qwen3-VL-8B-Instruct-unsloth-bnb-4bit` |

### Training Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--hf_dataset` | None | HuggingFace dataset (e.g., `Qhuy204/VQA`) |
| `--epochs` | 3 | Number of epochs |
| `--batch_size` | 2 | Batch size per GPU |
| `--lr` | 2e-4 | Learning rate |
| `--lora_r` | 16 | LoRA rank |

### A100 Optimized

```bash
python finetune_qwen3vl.py \
    --hf_dataset Qhuy204/VQA \
    --model_name unsloth/Qwen3-VL-8B-Instruct-unsloth-bnb-4bit \
    --batch_size 4 \
    --lora_r 32
```

## 📁 Project Structure

```
VQA/
├── finetune_qwen3vl.py     # Main training script
├── inference.py            # Inference script
├── requirements.txt        # Dependencies
├── configs/
│   ├── training_config.yaml      # Default config
│   └── training_config_a100.yaml # A100 optimized
└── README.md
```

## 📚 References

- [Unsloth Qwen3-VL Guide](https://unsloth.ai/docs/models/qwen3-vl-how-to-run-and-fine-tune)
- [Qwen3-VL Model](https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct)
- [VietTravelVQA Dataset](https://huggingface.co/datasets/Qhuy204/VQA)

## 📄 License

- **Code:** MIT License
- **Dataset:** CC BY 4.0
