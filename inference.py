"""
Inference script for finetuned Qwen3-VL model on VietTravelVQA
Using Unsloth FastVisionModel

Based on: https://unsloth.ai/docs/models/qwen3-vl-how-to-run-and-fine-tune

Usage:
    # Single image inference
    python inference.py --model_path outputs/qwen3vl-viettravelvqa/lora_model \
        --image VietTravelVQA/images/VN_000744.jpg \
        --question "Đây là công trình kiến trúc gì?"
    
    # Batch evaluation
    python inference.py --model_path outputs/qwen3vl-viettravelvqa/lora_model \
        --mode batch \
        --test_file VietTravelVQA/viettravelvqa_test.json \
        --max_samples 50
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Optional

import torch

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class VietTravelVQAInference:
    """
    Inference class for VietTravelVQA using Unsloth FastVisionModel
    """
    
    def __init__(
        self,
        model_path: str,
        base_model_name: str = "unsloth/Qwen3-VL-4B-Instruct-unsloth-bnb-4bit",
        load_in_4bit: bool = True,
    ):
        """
        Initialize inference engine
        
        Args:
            model_path: Path to LoRA adapter or full model
            base_model_name: Base model name (if loading LoRA)
            load_in_4bit: Whether to use 4-bit quantization
        """
        from unsloth import FastVisionModel
        from PIL import Image
        
        self.model_path = Path(model_path)
        self.Image = Image
        
        logger.info(f"Loading model from {model_path}")
        
        # Check if this is a LoRA adapter
        is_lora = (self.model_path / "adapter_config.json").exists()
        
        if is_lora:
            # Load LoRA adapter
            logger.info(f"Loading LoRA adapter from: {model_path}")
            self.model, self.tokenizer = FastVisionModel.from_pretrained(
                model_name=str(model_path),
                load_in_4bit=load_in_4bit,
            )
        else:
            # Load full model (fallback to base if path doesn't exist)
            model_to_load = str(model_path) if self.model_path.exists() else base_model_name
            logger.info(f"Loading model from: {model_to_load}")
            self.model, self.tokenizer = FastVisionModel.from_pretrained(
                model_name=model_to_load,
                load_in_4bit=load_in_4bit,
            )
        
        # Enable inference mode
        FastVisionModel.for_inference(self.model)
        
        logger.info("✅ Model loaded successfully!")
    
    def generate(
        self,
        image_path: str,
        question: str,
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.8,
        top_k: int = 20,
        stream: bool = False,
    ) -> str:
        """
        Generate answer for a given image and question
        
        Args:
            image_path: Path to input image
            question: Question in Vietnamese
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature (0.7 for Instruct, 1.0 for Thinking)
            top_p: Top-p sampling (0.8 for Instruct, 0.95 for Thinking)
            top_k: Top-k sampling
            stream: Whether to stream output
            
        Returns:
            Generated answer string
        """
        from transformers import TextStreamer
        
        # Load image
        image = self.Image.open(image_path).convert("RGB")
        
        # Create message in Qwen3-VL format
        messages = [
            {"role": "user", "content": [
                {"type": "image"},
                {"type": "text", "text": f"Câu hỏi: {question}"}
            ]}
        ]
        
        # Apply chat template
        input_text = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True
        )
        
        # Tokenize with image
        inputs = self.tokenizer(
            image,
            input_text,
            add_special_tokens=False,
            return_tensors="pt",
        ).to("cuda")
        
        # Setup streamer if needed
        streamer = None
        if stream:
            streamer = TextStreamer(self.tokenizer, skip_prompt=True)
        
        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                streamer=streamer,
                max_new_tokens=max_new_tokens,
                use_cache=True,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                min_p=0.1,
            )
        
        # Decode (skip input tokens)
        input_len = inputs["input_ids"].shape[1]
        generated_ids = outputs[0][input_len:]
        answer = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
        
        return answer.strip()
    
    def evaluate_batch(
        self,
        json_file: str,
        image_dir: str,
        max_samples: Optional[int] = None,
        output_file: Optional[str] = None,
    ) -> dict:
        """
        Evaluate model on VietTravelVQA test set
        
        Args:
            json_file: Path to test JSON file
            image_dir: Path to images directory
            max_samples: Maximum samples to evaluate
            output_file: Path to save predictions
            
        Returns:
            Dictionary with evaluation results
        """
        image_dir = Path(image_dir)
        
        # Load test data
        with open(json_file, "r", encoding="utf-8") as f:
            test_data = json.load(f)
        
        # Flatten QA pairs
        samples = []
        for entry in test_data:
            image_id = entry["image_id"]
            image_path = image_dir / image_id
            
            if not image_path.exists():
                continue
            
            for qa in entry["qa_pairs"]:
                samples.append({
                    "image_path": str(image_path),
                    "image_id": image_id,
                    "qid": qa["qid"],
                    "question": qa["question"],
                    "ground_truth": qa["answer"],
                    "difficulty": qa.get("difficulty", "3"),
                })
        
        if max_samples is not None:
            samples = samples[:max_samples]
        
        logger.info(f"Evaluating on {len(samples)} samples...")
        
        results = []
        for i, sample in enumerate(samples):
            if (i + 1) % 10 == 0:
                logger.info(f"[{i+1}/{len(samples)}] Processing...")
            
            try:
                prediction = self.generate(
                    image_path=sample["image_path"],
                    question=sample["question"],
                )
            except Exception as e:
                logger.warning(f"Error on {sample['qid']}: {e}")
                prediction = f"Error: {e}"
            
            results.append({
                **sample,
                "prediction": prediction,
            })
            
            # Print sample output for first few
            if i < 3:
                logger.info(f"  Q: {sample['question'][:80]}...")
                logger.info(f"  A: {prediction[:80]}...")
        
        # Save results
        if output_file:
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
            logger.info(f"✅ Predictions saved to {output_file}")
        
        return {"total": len(results), "results": results}


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="Inference with finetuned Qwen3-VL")
    
    # Required arguments
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to finetuned model (LoRA adapter directory)"
    )
    
    # Mode
    parser.add_argument(
        "--mode",
        type=str,
        choices=["single", "batch"],
        default="single",
        help="Inference mode"
    )
    
    # Single inference arguments
    parser.add_argument("--image", type=str, help="Path to input image")
    parser.add_argument("--question", type=str, help="Question to ask")
    
    # Batch evaluation arguments
    parser.add_argument(
        "--test_file",
        type=str,
        default="./VietTravelVQA/viettravelvqa_test.json",
        help="Path to test JSON file"
    )
    parser.add_argument(
        "--image_dir",
        type=str,
        default="./VietTravelVQA/images",
        help="Path to images directory"
    )
    parser.add_argument("--max_samples", type=int, default=None, help="Max samples")
    parser.add_argument("--output", type=str, default=None, help="Output JSON file")
    
    # Model arguments
    parser.add_argument(
        "--base_model",
        type=str,
        default="unsloth/Qwen3-VL-4B-Instruct-unsloth-bnb-4bit",
        help="Base model name"
    )
    parser.add_argument("--no_4bit", action="store_true", help="Disable 4-bit")
    
    # Generation arguments
    parser.add_argument("--max_tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_p", type=float, default=0.8)
    parser.add_argument("--top_k", type=int, default=20)
    parser.add_argument("--stream", action="store_true", help="Stream output")
    
    args = parser.parse_args()
    
    # Initialize inference engine
    engine = VietTravelVQAInference(
        model_path=args.model_path,
        base_model_name=args.base_model,
        load_in_4bit=not args.no_4bit,
    )
    
    if args.mode == "single":
        if not args.image or not args.question:
            parser.error("--image and --question required for single mode")
        
        print("\n" + "=" * 60)
        print(f"Image: {args.image}")
        print(f"Question: {args.question}")
        print("-" * 60)
        
        answer = engine.generate(
            image_path=args.image,
            question=args.question,
            max_new_tokens=args.max_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
            stream=args.stream,
        )
        
        if not args.stream:
            print(f"Answer: {answer}")
        print("=" * 60)
        
    else:
        results = engine.evaluate_batch(
            json_file=args.test_file,
            image_dir=args.image_dir,
            max_samples=args.max_samples,
            output_file=args.output,
        )
        
        print(f"\n✅ Evaluated {results['total']} samples")


if __name__ == "__main__":
    main()
