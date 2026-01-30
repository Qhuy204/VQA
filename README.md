# Qwen3-VL Finetuning on VietTravelVQA

Fine-tune **Qwen3-VL-4B-Instruct** trên bộ dữ liệu **VietTravelVQA** sử dụng **Unsloth** với **4-bit quantization**.

📚 **Docs:** https://unsloth.ai/docs/models/qwen3-vl-how-to-run-and-fine-tune

## 📋 Yêu cầu hệ thống

- **GPU:** NVIDIA GPU với ≥8GB VRAM (4B model), ≥16GB (8B model)
- **Python:** 3.10+
- **CUDA:** 11.8+ hoặc 12.x

## 🚀 Cài đặt

```bash
# 1. Tạo virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac

# 2. Cài đặt Unsloth
pip install unsloth

# 3. Cài đặt dependencies
pip install -r requirements.txt

# 4. (Optional) Flash Attention cho tốc độ tốt hơn
pip install flash-attn --no-build-isolation
```

## 📂 Cấu trúc thư mục

```
VQA/
├── VietTravelVQA/              # Dataset
│   ├── images/                 # 1406 images
│   ├── viettravelvqa_train.json  # 1124 images, 5620 QA pairs
│   └── viettravelvqa_test.json   # 282 images, 1410 QA pairs
├── configs/
│   └── training_config.yaml    # Training configuration
├── finetune_qwen3vl.py         # Main training script
├── inference.py                # Inference script
├── requirements.txt            # Dependencies
└── README.md
```

## 🎯 Training

### Quick Test (50 samples)

```bash
python finetune_qwen3vl.py --max_samples 50 --max_steps 30
```

### Full Training

```bash
python finetune_qwen3vl.py
```

### Custom Training

```bash
python finetune_qwen3vl.py \
    --model_name unsloth/Qwen3-VL-4B-Instruct-unsloth-bnb-4bit \
    --epochs 3 \
    --batch_size 2 \
    --grad_accum 4 \
    --lr 2e-4 \
    --lora_r 16 \
    --output_dir ./outputs/my_experiment
```

### Training Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--model_name` | `unsloth/Qwen3-VL-4B-Instruct-unsloth-bnb-4bit` | Model trên HuggingFace |
| `--epochs` | 3 | Số epochs |
| `--batch_size` | 2 | Batch size per GPU |
| `--grad_accum` | 4 | Gradient accumulation |
| `--lr` | 2e-4 | Learning rate |
| `--lora_r` | 16 | LoRA rank |
| `--lora_alpha` | 16 | LoRA alpha |
| `--max_steps` | -1 | Max steps (-1 = use epochs) |
| `--max_samples` | None | Limit samples (for testing) |
| `--save_gguf` | False | Export to GGUF format |

## 🔍 Inference

### Single Image

```bash
python inference.py \
    --model_path outputs/qwen3vl-viettravelvqa/lora_model \
    --image VietTravelVQA/images/VN_000744.jpg \
    --question "Đây là công trình kiến trúc gì?"
```

### With Streaming Output

```bash
python inference.py \
    --model_path outputs/qwen3vl-viettravelvqa/lora_model \
    --image VietTravelVQA/images/VN_000744.jpg \
    --question "Mô tả chi tiết hình ảnh này" \
    --stream
```

### Batch Evaluation

```bash
python inference.py \
    --model_path outputs/qwen3vl-viettravelvqa/lora_model \
    --mode batch \
    --test_file VietTravelVQA/viettravelvqa_test.json \
    --max_samples 100 \
    --output predictions.json
```

## 🧠 Model Architecture

### Unsloth Pre-quantized Models

| Model | VRAM | HuggingFace |
|-------|------|-------------|
| 2B | ~6GB | `unsloth/Qwen3-VL-2B-Instruct-unsloth-bnb-4bit` |
| **4B** | ~8GB | `unsloth/Qwen3-VL-4B-Instruct-unsloth-bnb-4bit` |
| 8B | ~12GB | `unsloth/Qwen3-VL-8B-Instruct-unsloth-bnb-4bit` |
| 32B | ~24GB | `unsloth/Qwen3-VL-32B-Instruct-unsloth-bnb-4bit` |

### LoRA Configuration

```python
model = FastVisionModel.get_peft_model(
    model,
    finetune_vision_layers=True,     # Vision encoder
    finetune_language_layers=True,   # LLM layers
    finetune_attention_modules=True, # Q, K, V, O
    finetune_mlp_modules=True,       # gate, up, down
    r=16,
    lora_alpha=16,
)
```

## 📊 Dataset Info

**VietTravelVQA** - VQA dataset về du lịch Việt Nam:

| Split | Images | QA Pairs | QA/Image |
|-------|--------|----------|----------|
| Train | 1,124 | 5,620 | 5 |
| Test | 282 | 1,410 | 5 |

**Difficulty Levels:**
1. **Very Easy** - Thuộc tính đơn giản (màu sắc, số lượng)
2. **Easy** - Suy luận cơ bản (đọc biển, vị trí)
3. **Medium** - Nhận dạng đối tượng (tên địa danh)
4. **Hard** - Suy luận ngữ cảnh (văn hóa, lịch sử)
5. **Very Hard** - Suy luận đa bước với kiến thức ngoài

## 💾 Export Models

### Save to GGUF (for llama.cpp)

```bash
python finetune_qwen3vl.py --save_gguf --gguf_quant q4_k_m
```

### Load and use GGUF

```bash
# Using llama.cpp
./llama-mtmd-cli \
    -hf unsloth/Qwen3-VL-4B-Instruct-GGUF:UD-Q4_K_XL \
    --n-gpu-layers 99 \
    --jinja \
    --top-p 0.8 --top-k 20 --temp 0.7
```

## 📝 Recommended Settings

### Instruct Model
- Temperature: 0.7
- Top-P: 0.8
- Top-K: 20
- Presence Penalty: 1.5

### Thinking Model
- Temperature: 1.0
- Top-P: 0.95
- Top-K: 20
- Presence Penalty: 0.0

## 🔗 References

- [Unsloth Qwen3-VL Guide](https://unsloth.ai/docs/models/qwen3-vl-how-to-run-and-fine-tune)
- [Colab Notebook](https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Qwen3_VL_(8B)-Vision.ipynb)
- [VietTravelVQA Dataset](./VietTravelVQA/README.md)

## 📜 License

- **Dataset:** CC BY 4.0
- **Images:** Creative Commons (Wikimedia Commons)
- **Code:** MIT License
