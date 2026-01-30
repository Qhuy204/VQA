"""
Finetuning Qwen3-VL 4B trên VietTravelVQA Dataset
Sử dụng Unsloth với 4-bit quantization

Dựa theo official docs: https://unsloth.ai/docs/models/qwen3-vl-how-to-run-and-fine-tune
Notebook ref: https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Qwen3_VL_(8B)-Vision.ipynb

Author: VietTravelVQA Team
Date: 2026-01-30
"""

import json
import os
import logging
from pathlib import Path
from typing import Optional

import torch

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("training.log")
    ]
)
logger = logging.getLogger(__name__)


# =============================================================================
# Installation Check
# =============================================================================

def check_dependencies():
    """Check and print dependency status"""
    try:
        from unsloth import FastVisionModel
        logger.info("✅ Unsloth FastVisionModel available")
        return True
    except ImportError:
        logger.error("❌ Unsloth not installed. Please run:")
        logger.error("   pip install unsloth")
        return False


# =============================================================================
# Dataset Preparation
# =============================================================================

def load_dataset_from_huggingface(
    repo_id: str = "Qhuy204/VQA",
    split: str = "train",
    max_samples: Optional[int] = None,
    cache_dir: Optional[str] = None,
) -> list:
    """
    Load VietTravelVQA dataset from HuggingFace Hub (Qhuy204/VQA)
    
    The dataset contains a ZIP file with:
    - images/ directory
    - viettravelvqa_train.json
    - viettravelvqa_test.json
    
    Args:
        repo_id: HuggingFace dataset repo ID (default: Qhuy204/VQA)
        split: Dataset split ("train" or "test")
        max_samples: Maximum number of samples
        cache_dir: Cache directory for downloaded files
        
    Returns:
        List of conversation samples in Unsloth format
    """
    from huggingface_hub import hf_hub_download
    from PIL import Image
    import zipfile
    import tempfile
    import shutil
    
    logger.info(f"Loading dataset from HuggingFace: {repo_id} (split: {split})")
    
    # Setup cache directory
    if cache_dir is None:
        cache_dir = os.path.join(tempfile.gettempdir(), "viettravelvqa_cache")
    
    extract_dir = os.path.join(cache_dir, "VietTravelVQA")
    
    # Check if already extracted
    json_file = os.path.join(extract_dir, f"viettravelvqa_{split}.json")
    image_dir = os.path.join(extract_dir, "images")
    
    if not os.path.exists(json_file):
        logger.info("Downloading and extracting dataset...")
        
        # Download ZIP file from HuggingFace
        zip_path = hf_hub_download(
            repo_id=repo_id,
            filename="VietTravelVQA.zip",
            repo_type="dataset",
            cache_dir=cache_dir,
        )
        
        logger.info(f"Downloaded to: {zip_path}")
        
        # Extract ZIP
        os.makedirs(cache_dir, exist_ok=True)
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            # Get the root folder name in the ZIP
            namelist = zip_ref.namelist()
            root_folder = namelist[0].split('/')[0] if namelist else ""
            
            logger.info(f"Extracting to: {cache_dir}")
            zip_ref.extractall(cache_dir)
        
        # Handle potential nested folder structure
        potential_paths = [
            os.path.join(cache_dir, "VietTravelVQA"),
            os.path.join(cache_dir, root_folder),
            cache_dir,
        ]
        
        for path in potential_paths:
            test_json = os.path.join(path, f"viettravelvqa_{split}.json")
            if os.path.exists(test_json):
                extract_dir = path
                json_file = test_json
                image_dir = os.path.join(path, "images")
                break
        
        logger.info(f"Extracted to: {extract_dir}")
    else:
        logger.info(f"Using cached dataset from: {extract_dir}")
    
    # Verify files exist
    if not os.path.exists(json_file):
        raise FileNotFoundError(f"JSON file not found: {json_file}")
    if not os.path.exists(image_dir):
        raise FileNotFoundError(f"Image directory not found: {image_dir}")
    
    # Load JSON annotations
    with open(json_file, "r", encoding="utf-8") as f:
        raw_data = json.load(f)
    
    logger.info(f"Loaded {len(raw_data)} entries from {json_file}")
    
    # Convert to Unsloth format
    samples = []
    image_dir_path = Path(image_dir)
    
    for entry in raw_data:
        image_id = entry["image_id"]
        image_path = image_dir_path / image_id
        
        if not image_path.exists():
            logger.warning(f"Image not found: {image_path}")
            continue
        
        try:
            image = Image.open(str(image_path)).convert("RGB")
        except Exception as e:
            logger.warning(f"Failed to load image {image_path}: {e}")
            continue
        
        for qa in entry["qa_pairs"]:
            question = qa.get("question", "")
            answer = qa.get("answer", "")
            
            if not question or not answer:
                continue
            
            conversation = {
                "messages": [
                    {"role": "user", "content": [
                        {"type": "image", "image": image},
                        {"type": "text", "text": f"Câu hỏi: {question}"}
                    ]},
                    {"role": "assistant", "content": [
                        {"type": "text", "text": answer}
                    ]}
                ]
            }
            samples.append(conversation)
            
            if max_samples is not None and len(samples) >= max_samples:
                break
        
        if max_samples is not None and len(samples) >= max_samples:
            break
    
    logger.info(f"✅ Loaded {len(samples)} QA pairs from HuggingFace ({repo_id})")
    return samples


class VietTravelVQADataset:
    """
    Load and convert VietTravelVQA dataset to Unsloth conversation format
    
    Supports:
    - Local JSON files
    - HuggingFace Hub (Qhuy204/VQA)
    
    Unsloth Vision format:
    {
        "messages": [
            {"role": "user", "content": [
                {"type": "text", "text": "..."},
                {"type": "image", "image": PIL.Image}
            ]},
            {"role": "assistant", "content": [
                {"type": "text", "text": "..."}
            ]}
        ]
    }
    """
    
    SYSTEM_PROMPT = (
        "Bạn là một trợ lý AI chuyên về du lịch Việt Nam. "
        "Hãy trả lời các câu hỏi về hình ảnh một cách chính xác và chi tiết."
    )
    
    def __init__(
        self,
        json_file: str,
        image_dir: str,
        max_samples: Optional[int] = None,
        include_system: bool = False,
    ):
        """
        Initialize dataset
        
        Args:
            json_file: Path to JSON annotation file
            image_dir: Path to images directory
            max_samples: Maximum number of samples (for debugging)
            include_system: Whether to include system prompt
        """
        from PIL import Image
        
        self.image_dir = Path(image_dir)
        self.include_system = include_system
        
        logger.info(f"Loading dataset from {json_file}")
        
        # Load annotations
        with open(json_file, "r", encoding="utf-8") as f:
            raw_data = json.load(f)
        
        # Flatten QA pairs and convert to conversation format
        self.samples = []
        for entry in raw_data:
            image_id = entry["image_id"]
            image_path = self.image_dir / image_id
            
            if not image_path.exists():
                logger.warning(f"Image not found: {image_path}")
                continue
            
            # Load image once for all QA pairs of this image
            try:
                image = Image.open(str(image_path)).convert("RGB")
            except Exception as e:
                logger.warning(f"Failed to load image {image_path}: {e}")
                continue
            
            for qa in entry["qa_pairs"]:
                conversation = self._create_conversation(
                    image=image,
                    question=qa["question"],
                    answer=qa["answer"],
                )
                self.samples.append(conversation)
        
        # Limit samples if specified
        if max_samples is not None:
            self.samples = self.samples[:max_samples]
        
        logger.info(f"✅ Loaded {len(self.samples)} QA pairs")
    
    def _create_conversation(self, image, question: str, answer: str) -> dict:
        """
        Create conversation in Unsloth format
        
        Args:
            image: PIL Image object
            question: Question text
            answer: Answer text
            
        Returns:
            Dictionary with 'messages' key in Unsloth format
        """
        messages = []
        
        # Optional system message
        if self.include_system:
            messages.append({
                "role": "system",
                "content": [{"type": "text", "text": self.SYSTEM_PROMPT}]
            })
        
        # User message with image and question
        messages.append({
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": f"Câu hỏi: {question}"}
            ]
        })
        
        # Assistant response
        messages.append({
            "role": "assistant",
            "content": [
                {"type": "text", "text": answer}
            ]
        })
        
        return {"messages": messages}
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> dict:
        return self.samples[idx]
    
    def to_list(self) -> list:
        """Return as list for SFTTrainer"""
        return self.samples


# =============================================================================
# Model Loading (Unsloth)
# =============================================================================

def load_model(
    model_name: str = "unsloth/Qwen3-VL-4B-Instruct-unsloth-bnb-4bit",
    load_in_4bit: bool = True,
    max_seq_length: int = 2048,
):
    """
    Load Qwen3-VL model using Unsloth FastVisionModel
    
    Args:
        model_name: HuggingFace model name
        load_in_4bit: Whether to use 4-bit quantization
        max_seq_length: Maximum sequence length
        
    Returns:
        tuple: (model, tokenizer)
    """
    from unsloth import FastVisionModel
    
    logger.info(f"Loading model: {model_name}")
    logger.info(f"4-bit quantization: {load_in_4bit}")
    
    model, tokenizer = FastVisionModel.from_pretrained(
        model_name,
        load_in_4bit=load_in_4bit,
        use_gradient_checkpointing="unsloth",  # Optimized for long context
    )
    
    logger.info("✅ Model loaded successfully")
    
    return model, tokenizer


def apply_lora(
    model,
    finetune_vision_layers: bool = True,
    finetune_language_layers: bool = True,
    finetune_attention_modules: bool = True,
    finetune_mlp_modules: bool = True,
    r: int = 16,
    lora_alpha: int = 16,
    lora_dropout: float = 0,
):
    """
    Apply LoRA adapters to model using Unsloth
    
    Args:
        model: Base model
        finetune_vision_layers: Whether to finetune vision encoder
        finetune_language_layers: Whether to finetune LLM layers
        finetune_attention_modules: Whether to finetune attention
        finetune_mlp_modules: Whether to finetune MLP layers
        r: LoRA rank
        lora_alpha: LoRA alpha
        lora_dropout: LoRA dropout
        
    Returns:
        Model with LoRA adapters
    """
    from unsloth import FastVisionModel
    
    logger.info("Applying LoRA adapters...")
    logger.info(f"  - Vision layers: {finetune_vision_layers}")
    logger.info(f"  - Language layers: {finetune_language_layers}")
    logger.info(f"  - Attention modules: {finetune_attention_modules}")
    logger.info(f"  - MLP modules: {finetune_mlp_modules}")
    logger.info(f"  - LoRA rank: {r}, alpha: {lora_alpha}")
    
    model = FastVisionModel.get_peft_model(
        model,
        finetune_vision_layers=finetune_vision_layers,
        finetune_language_layers=finetune_language_layers,
        finetune_attention_modules=finetune_attention_modules,
        finetune_mlp_modules=finetune_mlp_modules,
        r=r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        bias="none",
        random_state=3407,
        use_rslora=False,
        loftq_config=None,
    )
    
    logger.info("✅ LoRA adapters applied")
    
    return model


# =============================================================================
# Training
# =============================================================================

def train(
    model,
    tokenizer,
    train_dataset: list,
    eval_dataset: Optional[list] = None,
    output_dir: str = "./outputs/qwen3vl-viettravelvqa",
    num_train_epochs: int = 3,
    per_device_train_batch_size: int = 2,
    per_device_eval_batch_size: int = 2,
    gradient_accumulation_steps: int = 4,
    learning_rate: float = 2e-4,
    max_steps: int = -1,
    max_seq_length: int = 2048,
    warmup_steps: int = 5,
    logging_steps: int = 1,
    save_steps: int = 100,
    eval_steps: int = 100,
    seed: int = 3407,
):
    """
    Train model using SFTTrainer with Unsloth optimizations
    
    Args:
        model: Model with LoRA adapters
        tokenizer: Tokenizer
        train_dataset: List of conversation samples
        eval_dataset: Optional validation dataset for evaluation during training
        output_dir: Output directory for checkpoints
        num_train_epochs: Number of training epochs
        per_device_train_batch_size: Batch size per GPU for training
        per_device_eval_batch_size: Batch size per GPU for evaluation
        gradient_accumulation_steps: Gradient accumulation steps
        learning_rate: Learning rate
        max_steps: Max training steps (-1 for epochs-based)
        max_seq_length: Maximum sequence length
        warmup_steps: Warmup steps
        logging_steps: Logging frequency
        save_steps: Checkpoint save frequency
        eval_steps: Evaluation frequency (only used if eval_dataset provided)
        seed: Random seed
        
    Returns:
        Trainer statistics
    """
    from unsloth import FastVisionModel
    from unsloth.trainer import UnslothVisionDataCollator
    from trl import SFTTrainer, SFTConfig
    
    logger.info("=" * 60)
    logger.info("Starting Training")
    logger.info("=" * 60)
    logger.info(f"Training samples: {len(train_dataset)}")
    if eval_dataset:
        logger.info(f"Eval samples: {len(eval_dataset)}")
    logger.info(f"Batch size: {per_device_train_batch_size}")
    logger.info(f"Gradient accumulation: {gradient_accumulation_steps}")
    logger.info(f"Effective batch size: {per_device_train_batch_size * gradient_accumulation_steps}")
    logger.info(f"Learning rate: {learning_rate}")
    logger.info(f"Epochs: {num_train_epochs}")
    logger.info(f"Max steps: {max_steps}")
    if eval_dataset:
        logger.info(f"Eval every: {eval_steps} steps")
    
    # Enable training mode in Unsloth
    FastVisionModel.for_training(model)
    
    # Get initial GPU memory
    gpu_stats = torch.cuda.get_device_properties(0)
    start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
    max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
    logger.info(f"GPU: {gpu_stats.name}")
    logger.info(f"Max memory: {max_memory} GB")
    logger.info(f"Initial reserved: {start_gpu_memory} GB")
    
    # Determine evaluation strategy
    eval_strategy = "steps" if eval_dataset else "no"
    
    # Create trainer with Unsloth Vision DataCollator
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        data_collator=UnslothVisionDataCollator(model, tokenizer),  # Required for Vision
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        args=SFTConfig(
            per_device_train_batch_size=per_device_train_batch_size,
            per_device_eval_batch_size=per_device_eval_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            warmup_steps=warmup_steps,
            num_train_epochs=num_train_epochs,
            max_steps=max_steps,
            learning_rate=learning_rate,
            logging_steps=logging_steps,
            optim="adamw_8bit",
            weight_decay=0.01,
            lr_scheduler_type="linear",
            seed=seed,
            output_dir=output_dir,
            save_steps=save_steps,
            save_total_limit=3,
            eval_strategy=eval_strategy,
            eval_steps=eval_steps if eval_dataset else None,
            load_best_model_at_end=True if eval_dataset else False,
            metric_for_best_model="eval_loss" if eval_dataset else None,
            greater_is_better=False if eval_dataset else None,
            report_to="none",  # Set to "wandb" if needed
            
            # Required for Vision models
            remove_unused_columns=False,
            dataset_text_field="",
            dataset_kwargs={"skip_prepare_dataset": True},
            max_seq_length=max_seq_length,
        ),
    )
    
    # Train
    logger.info("Starting training loop...")
    trainer_stats = trainer.train()
    
    # Print memory stats
    used_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
    used_memory_for_lora = round(used_memory - start_gpu_memory, 3)
    used_percentage = round(used_memory / max_memory * 100, 3)
    lora_percentage = round(used_memory_for_lora / max_memory * 100, 3)
    
    logger.info("=" * 60)
    logger.info("Training Complete!")
    logger.info("=" * 60)
    logger.info(f"Training time: {trainer_stats.metrics['train_runtime']:.2f} seconds")
    logger.info(f"Training time: {trainer_stats.metrics['train_runtime']/60:.2f} minutes")
    logger.info(f"Peak reserved memory: {used_memory} GB")
    logger.info(f"Memory for LoRA training: {used_memory_for_lora} GB")
    logger.info(f"Peak memory %: {used_percentage}%")
    logger.info(f"LoRA memory %: {lora_percentage}%")
    
    return trainer, trainer_stats


def save_model(model, tokenizer, output_dir: str = "./outputs/qwen3vl-viettravelvqa"):
    """
    Save model and tokenizer
    
    Args:
        model: Trained model
        tokenizer: Tokenizer
        output_dir: Output directory
    """
    lora_path = os.path.join(output_dir, "lora_model")
    logger.info(f"Saving LoRA adapters to {lora_path}")
    
    model.save_pretrained(lora_path)
    tokenizer.save_pretrained(lora_path)
    
    logger.info("✅ Model saved successfully!")


def save_to_gguf(model, tokenizer, output_dir: str, quantization: str = "q4_k_m"):
    """
    Save model to GGUF format for llama.cpp
    
    Args:
        model: Trained model
        tokenizer: Tokenizer
        output_dir: Output directory
        quantization: Quantization method (q4_k_m, q8_0, etc.)
    """
    gguf_path = os.path.join(output_dir, "gguf")
    logger.info(f"Saving to GGUF with {quantization} quantization")
    
    model.save_pretrained_gguf(gguf_path, tokenizer, quantization_method=quantization)
    
    logger.info("✅ GGUF model saved successfully!")


# =============================================================================
# Inference
# =============================================================================

def run_inference(model, tokenizer, image_path: str, question: str):
    """
    Run inference on a single image
    
    Args:
        model: Trained model
        tokenizer: Tokenizer
        image_path: Path to image
        question: Question to ask
        
    Returns:
        Generated answer
    """
    from unsloth import FastVisionModel
    from PIL import Image
    from transformers import TextStreamer
    
    # Enable inference mode
    FastVisionModel.for_inference(model)
    
    # Load image
    image = Image.open(image_path).convert("RGB")
    
    # Create message
    messages = [
        {"role": "user", "content": [
            {"type": "image"},
            {"type": "text", "text": f"Câu hỏi: {question}"}
        ]}
    ]
    
    # Apply chat template
    input_text = tokenizer.apply_chat_template(messages, add_generation_prompt=True)
    
    # Tokenize
    inputs = tokenizer(
        image,
        input_text,
        add_special_tokens=False,
        return_tensors="pt",
    ).to("cuda")
    
    # Generate
    text_streamer = TextStreamer(tokenizer, skip_prompt=True)
    outputs = model.generate(
        **inputs,
        streamer=text_streamer,
        max_new_tokens=512,
        use_cache=True,
        temperature=0.7,
        top_p=0.8,
        top_k=20,
    )
    
    # Decode (without streamer for return value)
    generated_ids = outputs[0][inputs["input_ids"].shape[1]:]
    answer = tokenizer.decode(generated_ids, skip_special_tokens=True)
    
    return answer


# =============================================================================
# Main Entry Point
# =============================================================================

def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Finetune Qwen3-VL on VietTravelVQA")
    
    # Model arguments
    parser.add_argument(
        "--model_name",
        type=str,
        default="unsloth/Qwen3-VL-4B-Instruct-unsloth-bnb-4bit",
        help="Model name on HuggingFace"
    )
    parser.add_argument("--no_4bit", action="store_true", help="Disable 4-bit quantization")
    
    # LoRA arguments
    parser.add_argument("--lora_r", type=int, default=16, help="LoRA rank")
    parser.add_argument("--lora_alpha", type=int, default=16, help="LoRA alpha")
    parser.add_argument("--no_vision_finetune", action="store_true", help="Don't finetune vision layers")
    
    # Data arguments
    parser.add_argument(
        "--hf_dataset",
        type=str,
        default=None,
        help="HuggingFace dataset repo ID (e.g., Qhuy204/VQA). If set, ignores local files."
    )
    parser.add_argument(
        "--hf_split",
        type=str,
        default="train",
        help="HuggingFace dataset split (default: train)"
    )
    parser.add_argument(
        "--train_file",
        type=str,
        default="./VietTravelVQA/viettravelvqa_train.json",
        help="Training data JSON file (local)"
    )
    parser.add_argument(
        "--image_dir",
        type=str,
        default="./VietTravelVQA/images",
        help="Images directory (local)"
    )
    parser.add_argument("--max_samples", type=int, default=None, help="Max training samples")
    
    # Training arguments
    parser.add_argument("--output_dir", type=str, default="./outputs/qwen3vl-viettravelvqa")
    parser.add_argument("--epochs", type=int, default=3, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=2, help="Batch size per GPU")
    parser.add_argument("--grad_accum", type=int, default=4, help="Gradient accumulation steps")
    parser.add_argument("--lr", type=float, default=2e-4, help="Learning rate")
    parser.add_argument("--max_steps", type=int, default=-1, help="Max steps (-1 for epochs)")
    parser.add_argument("--max_seq_length", type=int, default=2048, help="Max sequence length")
    parser.add_argument("--warmup_steps", type=int, default=5, help="Warmup steps")
    parser.add_argument("--logging_steps", type=int, default=1, help="Logging frequency")
    parser.add_argument("--save_steps", type=int, default=100, help="Save checkpoint frequency")
    parser.add_argument("--eval_steps", type=int, default=100, help="Evaluation frequency")
    parser.add_argument(
        "--test_file",
        type=str,
        default="./VietTravelVQA/viettravelvqa_test.json",
        help="Test/validation data JSON file (local)"
    )
    parser.add_argument("--max_eval_samples", type=int, default=None, help="Max eval samples")
    parser.add_argument("--no_eval", action="store_true", help="Disable evaluation during training")
    
    # Output arguments
    parser.add_argument("--save_gguf", action="store_true", help="Also save to GGUF format")
    parser.add_argument("--gguf_quant", type=str, default="q4_k_m", help="GGUF quantization method")
    
    args = parser.parse_args()
    
    # Check dependencies
    if not check_dependencies():
        return
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load model
    model, tokenizer = load_model(
        model_name=args.model_name,
        load_in_4bit=not args.no_4bit,
        max_seq_length=args.max_seq_length,
    )
    
    # Apply LoRA
    model = apply_lora(
        model,
        finetune_vision_layers=not args.no_vision_finetune,
        finetune_language_layers=True,
        finetune_attention_modules=True,
        finetune_mlp_modules=True,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
    )
    
    # Load dataset (HuggingFace or local)
    if args.hf_dataset:
        # Load from HuggingFace Hub
        logger.info(f"Using HuggingFace dataset: {args.hf_dataset}")
        train_samples = load_dataset_from_huggingface(
            repo_id=args.hf_dataset,
            split=args.hf_split,
            max_samples=args.max_samples,
        )
    else:
        # Load from local files
        logger.info("Using local dataset files")
        dataset = VietTravelVQADataset(
            json_file=args.train_file,
            image_dir=args.image_dir,
            max_samples=args.max_samples,
        )
        train_samples = dataset.to_list()
    
    # Load eval dataset (if not disabled)
    eval_samples = None
    if not args.no_eval:
        if args.hf_dataset:
            # Load test split from HuggingFace
            logger.info("Loading eval dataset from HuggingFace (test split)...")
            eval_samples = load_dataset_from_huggingface(
                repo_id=args.hf_dataset,
                split="test",
                max_samples=args.max_eval_samples,
            )
        elif os.path.exists(args.test_file):
            # Load from local test file
            logger.info(f"Loading eval dataset from {args.test_file}...")
            eval_dataset_obj = VietTravelVQADataset(
                json_file=args.test_file,
                image_dir=args.image_dir,
                max_samples=args.max_eval_samples,
            )
            eval_samples = eval_dataset_obj.to_list()
        
        if eval_samples:
            logger.info(f"✅ Loaded {len(eval_samples)} eval samples")
        else:
            logger.warning("No eval dataset found, training without evaluation")
    
    # Train
    trainer, trainer_stats = train(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_samples,
        eval_dataset=eval_samples,
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        max_steps=args.max_steps,
        max_seq_length=args.max_seq_length,
        warmup_steps=args.warmup_steps,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        eval_steps=args.eval_steps,
    )
    
    # Save model
    save_model(model, tokenizer, args.output_dir)
    
    # Optionally save to GGUF
    if args.save_gguf:
        save_to_gguf(model, tokenizer, args.output_dir, args.gguf_quant)
    
    logger.info("🎉 Finetuning complete!")


if __name__ == "__main__":
    main()
