"""
evaluate.py

Standalone evaluation script for pre/post training baselines.
Uses same reward logic as training.
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from dataset_generator import generate_dataset
from reward import compute_alignment_score, parse_output

# ============================================================================
# CONFIG
# ============================================================================

MODEL_NAME = "meta-llama/Llama-3.2-3B-Instruct"
NUM_EVAL_SAMPLES = 100  # subset of test set
MAX_NEW_TOKENS = 256

# ============================================================================
# EVALUATION
# ============================================================================

def generate_completion(model, tokenizer, prompt, max_new_tokens=MAX_NEW_TOKENS):
    """Generate a single completion."""
    inputs = tokenizer.apply_chat_template(
        prompt, 
        return_tensors="pt",
        add_generation_prompt=True
    ).to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,  # greedy for reproducibility
            pad_token_id=tokenizer.eos_token_id,
        )
    
    # Decode only the new tokens
    completion = tokenizer.decode(outputs[0][inputs.shape[1]:], skip_special_tokens=True)
    return completion


def run_eval(model, tokenizer, dataset, num_samples=NUM_EVAL_SAMPLES, verbose=False, verbose_rate=10):
    """Run evaluation on dataset, return scores."""
    num_samples = min(num_samples, len(dataset))
    scores = []
    
    for i in range(num_samples):
        sample = dataset[i]
        completion = generate_completion(model, tokenizer, sample["prompt"])
        score = compute_alignment_score(sample["expected"], parse_output(completion))
        scores.append(score)
        
        # Show verbose output for every Nth sample
        if verbose and (i % verbose_rate == 0):
            print()
            print(f"{'='*60}")
            print(f"Sample {i+1}/{num_samples} - Score: {score:.3f}")
            print(f"{'='*60}")
            print(f"Fragment A: {sample['fragment_a'][:100]}...")
            print(f"Fragment B: {sample['fragment_b'][:100]}...")
            print()
            print(f"Expected ({len(sample['expected'])} words):")
            print("  " + " ".join(sample['expected'][:20]) + ("..." if len(sample['expected']) > 20 else ""))
            print()
            output_words = parse_output(completion)
            print(f"Model Output ({len(output_words)} words):")
            print("  " + " ".join(output_words[:20]) + ("..." if len(output_words) > 20 else ""))
            print()
            
        if (i + 1) % 10 == 0:
            print(f"  {i+1}/{num_samples} - running avg: {sum(scores)/len(scores):.3f}")
    
    return scores


# ============================================================================
# CLI
# ============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Evaluate interleaving performance")
    parser.add_argument("--model", default=MODEL_NAME, help="Model name or path")
    parser.add_argument("--dataset", default=None, help="Path to JSONL dataset")
    parser.add_argument("--samples", type=int, default=NUM_EVAL_SAMPLES, help="Number of samples")
    parser.add_argument("--verbose", action="store_true", help="Show individual completions")
    parser.add_argument("--verbose-rate", type=int, default=10, help="Show verbose output every N samples (default: 10)")
    args = parser.parse_args()
    
    print(f"Loading model: {args.model}")
    
    # For local checkpoints, load tokenizer from base model
    import os
    if os.path.isdir(args.model):
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    else:
        tokenizer = AutoTokenizer.from_pretrained(args.model)
    
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    
    if args.dataset:
        print(f"Loading dataset: {args.dataset}")
        _, _, test = generate_dataset(dataset_path=args.dataset)
    else:
        print("Generating dataset...")
        _, _, test = generate_dataset()
    
    print(f"Evaluating on {args.samples} samples...")
    scores = run_eval(model, tokenizer, test, args.samples, verbose=args.verbose, verbose_rate=args.verbose_rate)
    
    print()
    print("=" * 40)
    print(f"Results ({args.samples} samples):")
    print(f"  Mean score: {sum(scores)/len(scores):.3f}")
    print(f"  Min: {min(scores):.3f}")
    print(f"  Max: {max(scores):.3f}")
    print("=" * 40)