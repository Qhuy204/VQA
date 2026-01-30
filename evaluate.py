"""
Evaluation script for VietTravelVQA
Metrics: Accuracy, BLEU, ROUGE-L, F1

Usage:
    python evaluate.py \
        --predictions predictions.json \
        --output evaluation_results.json
        
    # Or evaluate directly from model
    python evaluate.py \
        --model_path outputs/qwen3vl-viettravelvqa/lora_model \
        --test_file VietTravelVQA/viettravelvqa_test.json \
        --image_dir VietTravelVQA/images \
        --max_samples 100
"""

import json
import re
import argparse
import logging
from pathlib import Path
from typing import Optional
from collections import Counter

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


# =============================================================================
# Text Preprocessing
# =============================================================================

def normalize_text(text: str) -> str:
    """Normalize Vietnamese text for comparison"""
    text = text.lower().strip()
    # Remove punctuation
    text = re.sub(r'[.,!?;:"\'\(\)\[\]{}]', '', text)
    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text)
    return text


def tokenize(text: str) -> list:
    """Simple whitespace tokenization"""
    return normalize_text(text).split()


# =============================================================================
# Metrics
# =============================================================================

def exact_match(prediction: str, ground_truth: str) -> float:
    """Exact match accuracy (after normalization)"""
    return 1.0 if normalize_text(prediction) == normalize_text(ground_truth) else 0.0


def contains_match(prediction: str, ground_truth: str) -> float:
    """Check if ground truth is contained in prediction"""
    pred_norm = normalize_text(prediction)
    gt_norm = normalize_text(ground_truth)
    return 1.0 if gt_norm in pred_norm or pred_norm in gt_norm else 0.0


def f1_score(prediction: str, ground_truth: str) -> float:
    """Token-level F1 score"""
    pred_tokens = tokenize(prediction)
    gt_tokens = tokenize(ground_truth)
    
    if not pred_tokens or not gt_tokens:
        return 0.0
    
    pred_counter = Counter(pred_tokens)
    gt_counter = Counter(gt_tokens)
    
    # Count common tokens
    common = sum((pred_counter & gt_counter).values())
    
    if common == 0:
        return 0.0
    
    precision = common / len(pred_tokens)
    recall = common / len(gt_tokens)
    f1 = 2 * precision * recall / (precision + recall)
    
    return f1


def bleu_score(prediction: str, ground_truth: str, max_n: int = 4) -> float:
    """
    Calculate BLEU score (simplified version)
    
    Args:
        prediction: Model prediction
        ground_truth: Ground truth answer
        max_n: Maximum n-gram to consider
        
    Returns:
        BLEU score (0-1)
    """
    pred_tokens = tokenize(prediction)
    gt_tokens = tokenize(ground_truth)
    
    if not pred_tokens or not gt_tokens:
        return 0.0
    
    # Calculate n-gram precisions
    precisions = []
    for n in range(1, min(max_n + 1, len(pred_tokens) + 1)):
        pred_ngrams = [tuple(pred_tokens[i:i+n]) for i in range(len(pred_tokens) - n + 1)]
        gt_ngrams = [tuple(gt_tokens[i:i+n]) for i in range(len(gt_tokens) - n + 1)]
        
        if not pred_ngrams:
            continue
        
        pred_counter = Counter(pred_ngrams)
        gt_counter = Counter(gt_ngrams)
        
        clipped = sum((pred_counter & gt_counter).values())
        total = sum(pred_counter.values())
        
        if total > 0:
            precisions.append(clipped / total)
    
    if not precisions:
        return 0.0
    
    # Geometric mean of precisions
    import math
    log_precision = sum(math.log(p) if p > 0 else -float('inf') for p in precisions) / len(precisions)
    
    if log_precision == -float('inf'):
        return 0.0
    
    bleu = math.exp(log_precision)
    
    # Brevity penalty
    if len(pred_tokens) < len(gt_tokens):
        bp = math.exp(1 - len(gt_tokens) / len(pred_tokens))
        bleu *= bp
    
    return bleu


def rouge_l(prediction: str, ground_truth: str) -> float:
    """
    Calculate ROUGE-L score (Longest Common Subsequence)
    
    Returns:
        F1 score based on LCS
    """
    pred_tokens = tokenize(prediction)
    gt_tokens = tokenize(ground_truth)
    
    if not pred_tokens or not gt_tokens:
        return 0.0
    
    # LCS using dynamic programming
    m, n = len(pred_tokens), len(gt_tokens)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if pred_tokens[i-1] == gt_tokens[j-1]:
                dp[i][j] = dp[i-1][j-1] + 1
            else:
                dp[i][j] = max(dp[i-1][j], dp[i][j-1])
    
    lcs_length = dp[m][n]
    
    if lcs_length == 0:
        return 0.0
    
    precision = lcs_length / m
    recall = lcs_length / n
    f1 = 2 * precision * recall / (precision + recall)
    
    return f1


# =============================================================================
# Evaluation
# =============================================================================

def evaluate_predictions(predictions: list) -> dict:
    """
    Evaluate predictions against ground truth
    
    Args:
        predictions: List of dicts with 'prediction' and 'ground_truth' keys
        
    Returns:
        Dictionary with evaluation metrics
    """
    metrics = {
        "exact_match": [],
        "contains_match": [],
        "f1": [],
        "bleu": [],
        "rouge_l": [],
    }
    
    for item in predictions:
        pred = item.get("prediction", "")
        gt = item.get("ground_truth", item.get("answer", ""))
        
        if not pred or not gt:
            continue
        
        metrics["exact_match"].append(exact_match(pred, gt))
        metrics["contains_match"].append(contains_match(pred, gt))
        metrics["f1"].append(f1_score(pred, gt))
        metrics["bleu"].append(bleu_score(pred, gt))
        metrics["rouge_l"].append(rouge_l(pred, gt))
    
    # Calculate averages
    results = {
        "num_samples": len(predictions),
        "num_evaluated": len(metrics["exact_match"]),
    }
    
    for name, values in metrics.items():
        if values:
            results[name] = sum(values) / len(values)
        else:
            results[name] = 0.0
    
    return results


def evaluate_by_difficulty(predictions: list) -> dict:
    """Evaluate predictions grouped by difficulty level"""
    by_difficulty = {}
    
    for item in predictions:
        diff = str(item.get("difficulty", "unknown"))
        if diff not in by_difficulty:
            by_difficulty[diff] = []
        by_difficulty[diff].append(item)
    
    results = {}
    for diff, items in sorted(by_difficulty.items()):
        results[f"difficulty_{diff}"] = evaluate_predictions(items)
    
    return results


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Evaluate VietTravelVQA predictions")
    
    # Input options
    parser.add_argument("--predictions", type=str, help="Path to predictions JSON file")
    parser.add_argument("--model_path", type=str, help="Path to model for direct evaluation")
    parser.add_argument("--test_file", type=str, default="./VietTravelVQA/viettravelvqa_test.json")
    parser.add_argument("--image_dir", type=str, default="./VietTravelVQA/images")
    parser.add_argument("--hf_dataset", type=str, help="HuggingFace dataset ID")
    
    # Options
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--output", type=str, default="evaluation_results.json")
    parser.add_argument("--by_difficulty", action="store_true", help="Also evaluate by difficulty")
    
    args = parser.parse_args()
    
    if args.predictions:
        # Load existing predictions
        logger.info(f"Loading predictions from {args.predictions}")
        with open(args.predictions, "r", encoding="utf-8") as f:
            predictions = json.load(f)
    elif args.model_path:
        # Generate predictions using model
        logger.info(f"Generating predictions using model: {args.model_path}")
        from inference import VietTravelVQAInference
        
        engine = VietTravelVQAInference(model_path=args.model_path)
        
        if args.hf_dataset:
            from finetune_qwen3vl import load_dataset_from_huggingface
            # For HF, we need to load test split
            # This is a simplified version - full implementation would need test data
            logger.warning("HuggingFace evaluation not fully implemented yet")
            return
        
        results = engine.evaluate_batch(
            json_file=args.test_file,
            image_dir=args.image_dir,
            max_samples=args.max_samples,
            output_file=None,
        )
        predictions = results["results"]
    else:
        parser.error("Either --predictions or --model_path must be specified")
        return
    
    # Evaluate
    logger.info(f"Evaluating {len(predictions)} predictions...")
    
    results = {
        "overall": evaluate_predictions(predictions)
    }
    
    if args.by_difficulty:
        results["by_difficulty"] = evaluate_by_difficulty(predictions)
    
    # Print results
    print("\n" + "=" * 60)
    print("📊 EVALUATION RESULTS")
    print("=" * 60)
    print(f"\nTotal samples: {results['overall']['num_samples']}")
    print(f"Evaluated: {results['overall']['num_evaluated']}")
    print()
    print(f"{'Metric':<20} {'Score':>10}")
    print("-" * 32)
    print(f"{'Exact Match':<20} {results['overall']['exact_match']*100:>9.2f}%")
    print(f"{'Contains Match':<20} {results['overall']['contains_match']*100:>9.2f}%")
    print(f"{'F1 Score':<20} {results['overall']['f1']*100:>9.2f}%")
    print(f"{'BLEU':<20} {results['overall']['bleu']*100:>9.2f}%")
    print(f"{'ROUGE-L':<20} {results['overall']['rouge_l']*100:>9.2f}%")
    
    if args.by_difficulty and "by_difficulty" in results:
        print("\n" + "-" * 60)
        print("📈 By Difficulty Level")
        print("-" * 60)
        for diff, metrics in results["by_difficulty"].items():
            print(f"\n{diff} (n={metrics['num_evaluated']}):")
            print(f"  Exact Match: {metrics['exact_match']*100:.2f}%")
            print(f"  F1: {metrics['f1']*100:.2f}%")
    
    print("\n" + "=" * 60)
    
    # Save results
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    logger.info(f"✅ Results saved to {args.output}")


if __name__ == "__main__":
    main()
