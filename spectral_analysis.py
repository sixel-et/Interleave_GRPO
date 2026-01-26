"""
spectral_analysis.py

Compute representation geometry metrics (RankMe, αReQ) for interleaving task.
Compares sequential vs interleaving prompt conditions across checkpoints.

Usage:
    # Compare base model conditions
    python spectral_analysis.py --model meta-llama/Llama-3.2-3B-Instruct --dataset datasets/10words.jsonl
    
    # Compare trained checkpoint
    python spectral_analysis.py --model outputs/Llama-3B-interleave/checkpoint-500 --dataset datasets/10words.jsonl
    
    # Run on multiple curriculum stages
    python spectral_analysis.py --model outputs/Llama-3B-interleave/checkpoint-500 --curriculum-dir datasets/
"""

import argparse
import json
import torch
import numpy as np
from pathlib import Path
from datetime import datetime
from transformers import AutoModelForCausalLM, AutoTokenizer
from dataset_generator import generate_dataset, load_jsonl

# ============================================================================
# CONFIG
# ============================================================================

BASE_MODEL = "meta-llama/Llama-3.2-3B-Instruct"
NUM_SAMPLES = 500
BOOTSTRAP_ITERATIONS = 1000
BOOTSTRAP_CI = 0.95

# ============================================================================
# SPECTRAL METRICS
# ============================================================================

def compute_covariance(hidden_states: torch.Tensor) -> torch.Tensor:
    """
    Compute covariance matrix from hidden states.
    
    Args:
        hidden_states: (N, d) tensor of hidden state vectors
    
    Returns:
        (d, d) covariance matrix
    """
    # Center
    centered = hidden_states - hidden_states.mean(dim=0, keepdim=True)
    # Covariance
    cov = (centered.T @ centered) / (hidden_states.shape[0] - 1)
    return cov


def compute_rankme(eigenvalues: torch.Tensor, eps: float = 1e-10) -> float:
    """
    Compute RankMe (effective rank) from eigenvalues.
    
    RankMe = exp(entropy of normalized eigenvalue distribution)
    """
    # Filter near-zero eigenvalues
    eigenvalues = eigenvalues[eigenvalues > eps]
    
    # Normalize to probability distribution
    p = eigenvalues / eigenvalues.sum()
    
    # Shannon entropy
    entropy = -(p * torch.log(p)).sum()
    
    # Effective rank
    rankme = torch.exp(entropy).item()
    return rankme


def compute_alpha_req(eigenvalues: torch.Tensor, eps: float = 1e-10) -> float:
    """
    Compute αReQ (power-law decay rate) from eigenvalues.
    
    Fits: log(σ_i) = -α * log(i) + constant
    """
    # Filter near-zero eigenvalues
    eigenvalues = eigenvalues[eigenvalues > eps]
    
    # Log-log linear regression
    log_eigenvalues = torch.log(eigenvalues).cpu().numpy()
    log_indices = np.log(np.arange(1, len(eigenvalues) + 1))
    
    # Fit line: slope is -α
    coeffs = np.polyfit(log_indices, log_eigenvalues, 1)
    alpha_req = -coeffs[0]
    
    return alpha_req


def compute_spectral_metrics(hidden_states: torch.Tensor) -> dict:
    """
    Compute all spectral metrics from hidden states.
    
    Args:
        hidden_states: (N, d) tensor
    
    Returns:
        dict with rankme, alpha_req, eigenvalues
    """
    cov = compute_covariance(hidden_states)
    
    # Eigendecomposition (eigenvalues in ascending order)
    eigenvalues = torch.linalg.eigvalsh(cov)
    eigenvalues = eigenvalues.flip(0)  # Descending order
    
    rankme = compute_rankme(eigenvalues)
    alpha_req = compute_alpha_req(eigenvalues)
    
    return {
        "rankme": rankme,
        "alpha_req": alpha_req,
        "eigenvalues": eigenvalues.cpu().numpy().tolist()
    }


def bootstrap_spectral_metrics(
    hidden_states: torch.Tensor,
    n_iterations: int = BOOTSTRAP_ITERATIONS,
    ci: float = BOOTSTRAP_CI,
    seed: int = 42
) -> dict:
    """
    Compute spectral metrics with bootstrap confidence intervals.
    """
    rng = np.random.RandomState(seed)
    n_samples = hidden_states.shape[0]
    
    rankme_samples = []
    alpha_req_samples = []
    
    for _ in range(n_iterations):
        # Sample with replacement
        indices = rng.choice(n_samples, size=n_samples, replace=True)
        sampled = hidden_states[indices]
        
        metrics = compute_spectral_metrics(sampled)
        rankme_samples.append(metrics["rankme"])
        alpha_req_samples.append(metrics["alpha_req"])
    
    # Compute CIs
    lower_pct = (1 - ci) / 2 * 100
    upper_pct = (1 + ci) / 2 * 100
    
    # Point estimates from full data
    full_metrics = compute_spectral_metrics(hidden_states)
    
    return {
        "rankme": full_metrics["rankme"],
        "rankme_ci": [
            np.percentile(rankme_samples, lower_pct),
            np.percentile(rankme_samples, upper_pct)
        ],
        "alpha_req": full_metrics["alpha_req"],
        "alpha_req_ci": [
            np.percentile(alpha_req_samples, lower_pct),
            np.percentile(alpha_req_samples, upper_pct)
        ],
        "eigenvalues": full_metrics["eigenvalues"]
    }


# ============================================================================
# HIDDEN STATE COLLECTION
# ============================================================================

def collect_hidden_states(
    model,
    tokenizer,
    texts: list[str],
    batch_size: int = 8,
    show_progress: bool = True
) -> torch.Tensor:
    """
    Collect last-layer, last-token hidden states for a list of texts.
    
    Args:
        model: HuggingFace model
        tokenizer: HuggingFace tokenizer
        texts: list of strings to process
        batch_size: batch size for forward passes
    
    Returns:
        (N, hidden_dim) tensor of hidden states
    """
    model.eval()
    hidden_states_list = []
    
    n_batches = (len(texts) + batch_size - 1) // batch_size
    
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i + batch_size]
        
        # Tokenize
        inputs = tokenizer(
            batch_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=2048
        ).to(model.device)
        
        # Forward pass
        with torch.no_grad():
            outputs = model(
                **inputs,
                output_hidden_states=True,
                return_dict=True
            )
        
        # Extract last layer hidden states
        last_layer = outputs.hidden_states[-1]  # (batch, seq_len, hidden_dim)
        
        # Get last non-padding token for each sequence
        attention_mask = inputs["attention_mask"]
        seq_lengths = attention_mask.sum(dim=1) - 1  # -1 for 0-indexing
        
        for j, seq_len in enumerate(seq_lengths):
            hidden_state = last_layer[j, seq_len, :]
            hidden_states_list.append(hidden_state)
        
        if show_progress and (i // batch_size + 1) % 10 == 0:
            print(f"  Batch {i // batch_size + 1}/{n_batches}")
    
    return torch.stack(hidden_states_list)


# ============================================================================
# CONDITION BUILDERS
# ============================================================================

def build_sequential_texts(samples: list[dict]) -> list[str]:
    """Build sequential (concatenated) condition texts."""
    texts = []
    for sample in samples:
        text = f"{sample['fragment_a']} {sample['fragment_b']}"
        texts.append(text)
    return texts


def build_interleave_prompt_texts(samples: list[dict], tokenizer) -> list[str]:
    """Build interleaving prompt condition texts."""
    texts = []
    for sample in samples:
        # Apply chat template to get the full prompt string
        prompt_text = tokenizer.apply_chat_template(
            sample["prompt"],
            tokenize=False,
            add_generation_prompt=True
        )
        texts.append(prompt_text)
    return texts


# ============================================================================
# MAIN ANALYSIS
# ============================================================================

def analyze_checkpoint(
    model_path: str,
    dataset_path: str,
    num_samples: int = NUM_SAMPLES,
    batch_size: int = 8,
    bootstrap: bool = True
) -> dict:
    """
    Run spectral analysis on a checkpoint for both conditions.
    
    Returns dict with results for each condition.
    """
    print(f"\n{'='*60}")
    print(f"Analyzing: {model_path}")
    print(f"Dataset: {dataset_path}")
    print(f"{'='*60}")
    
    # Load tokenizer (always from base model for consistency)
    print("\nLoading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    tokenizer.pad_token = tokenizer.eos_token
    
    # Load model
    print(f"Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        output_hidden_states=True
    )
    print(f"  Parameters: {model.num_parameters():,}")
    print(f"  Hidden dim: {model.config.hidden_size}")
    
    # Load dataset
    print(f"\nLoading dataset...")
    samples = load_jsonl(dataset_path)
    num_words = samples[0].get("num_words", "unknown")
    print(f"  Total samples: {len(samples)}")
    print(f"  Fragment length: {num_words} words")
    
    # Subsample if needed
    if len(samples) > num_samples:
        samples = samples[:num_samples]
        print(f"  Using first {num_samples} samples")
    
    results = {
        "model_path": model_path,
        "dataset_path": dataset_path,
        "num_words": num_words,
        "num_samples": len(samples),
        "timestamp": datetime.now().isoformat(),
        "conditions": {}
    }
    
    # Condition 1: Sequential
    print(f"\nCondition 1: Sequential (concatenated fragments)")
    sequential_texts = build_sequential_texts(samples)
    print(f"  Example: {sequential_texts[0][:100]}...")
    
    print(f"  Collecting hidden states...")
    sequential_hidden = collect_hidden_states(model, tokenizer, sequential_texts, batch_size)
    print(f"  Shape: {sequential_hidden.shape}")
    
    print(f"  Computing metrics...")
    if bootstrap:
        sequential_metrics = bootstrap_spectral_metrics(sequential_hidden)
    else:
        sequential_metrics = compute_spectral_metrics(sequential_hidden)
    results["conditions"]["sequential"] = sequential_metrics
    
    # Condition 2: Interleave prompt
    print(f"\nCondition 2: Interleave prompt")
    interleave_texts = build_interleave_prompt_texts(samples, tokenizer)
    print(f"  Example (last 200 chars): ...{interleave_texts[0][-200:]}")
    
    print(f"  Collecting hidden states...")
    interleave_hidden = collect_hidden_states(model, tokenizer, interleave_texts, batch_size)
    print(f"  Shape: {interleave_hidden.shape}")
    
    print(f"  Computing metrics...")
    if bootstrap:
        interleave_metrics = bootstrap_spectral_metrics(interleave_hidden)
    else:
        interleave_metrics = compute_spectral_metrics(interleave_hidden)
    results["conditions"]["interleave_prompt"] = interleave_metrics
    
    return results


def print_results(results: dict):
    """Print results to console."""
    print(f"\n{'='*60}")
    print("RESULTS")
    print(f"{'='*60}")
    print(f"Model: {results['model_path']}")
    print(f"Dataset: {results['dataset_path']} ({results['num_words']} words)")
    print(f"Samples: {results['num_samples']}")
    print()
    
    header = f"{'Condition':<20} {'RankMe':>12} {'95% CI':>20} {'αReQ':>10} {'95% CI':>18}"
    print(header)
    print("-" * len(header))
    
    for cond_name, metrics in results["conditions"].items():
        rankme = metrics["rankme"]
        alpha = metrics["alpha_req"]
        
        if "rankme_ci" in metrics:
            rankme_ci = f"[{metrics['rankme_ci'][0]:.1f}, {metrics['rankme_ci'][1]:.1f}]"
            alpha_ci = f"[{metrics['alpha_req_ci'][0]:.2f}, {metrics['alpha_req_ci'][1]:.2f}]"
        else:
            rankme_ci = "N/A"
            alpha_ci = "N/A"
        
        print(f"{cond_name:<20} {rankme:>12.1f} {rankme_ci:>20} {alpha:>10.3f} {alpha_ci:>18}")


def save_results(results: dict, output_path: str):
    """Save results to JSON file."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nSaved results to: {output_path}")


# ============================================================================
# CLI
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Spectral analysis of interleaving task")
    parser.add_argument("--model", default=BASE_MODEL, help="Model name or checkpoint path")
    parser.add_argument("--dataset", required=True, help="Path to JSONL dataset")
    parser.add_argument("--samples", type=int, default=NUM_SAMPLES, help="Number of samples")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size for forward passes")
    parser.add_argument("--no-bootstrap", action="store_true", help="Skip bootstrap CIs")
    parser.add_argument("--output", default=None, help="Output JSON path (default: auto-generated)")
    
    # Curriculum mode
    parser.add_argument("--curriculum-dir", default=None, 
                        help="Directory with curriculum datasets (runs on all)")
    
    args = parser.parse_args()
    
    if args.curriculum_dir:
        # Run on all curriculum stages
        curriculum_dir = Path(args.curriculum_dir)
        datasets = sorted(curriculum_dir.glob("*words.jsonl"))
        
        if not datasets:
            print(f"No *words.jsonl files found in {curriculum_dir}")
            return
        
        print(f"Found {len(datasets)} curriculum stages")
        
        all_results = []
        for dataset_path in datasets:
            results = analyze_checkpoint(
                args.model,
                str(dataset_path),
                args.samples,
                args.batch_size,
                bootstrap=not args.no_bootstrap
            )
            print_results(results)
            all_results.append(results)
        
        # Save combined results
        if args.output:
            output_path = args.output
        else:
            model_name = Path(args.model).name if Path(args.model).exists() else args.model.replace("/", "_")
            output_path = f"spectral_results/{model_name}_curriculum.json"
        
        save_results({"runs": all_results}, output_path)
    
    else:
        # Single dataset
        results = analyze_checkpoint(
            args.model,
            args.dataset,
            args.samples,
            args.batch_size,
            bootstrap=not args.no_bootstrap
        )
        print_results(results)
        
        # Save results
        if args.output:
            output_path = args.output
        else:
            model_name = Path(args.model).name if Path(args.model).exists() else args.model.replace("/", "_")
            dataset_name = Path(args.dataset).stem
            output_path = f"spectral_results/{model_name}_{dataset_name}.json"
        
        save_results(results, output_path)


if __name__ == "__main__":
    main()