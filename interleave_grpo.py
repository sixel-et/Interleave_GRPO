"""
interleave_grpo.py
GRPO training for the interleaving task.
Based on Will Brown's GSM8K demo.

Usage:
    python interleave_grpo.py
    python interleave_grpo.py --resume  # Resume from latest checkpoint
"""

import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainerCallback
from trl import GRPOConfig, GRPOTrainer
from dataset_generator import generate_dataset
from reward import nw_align
import argparse
from datetime import datetime

# ============================================================================
# CONFIG - CHANGE THESE
# ============================================================================

MODEL_NAME = "meta-llama/Llama-3.2-3B-Instruct"
OUTPUT_DIR = "outputs/Llama-3B-interleave"
RUN_NAME = "Llama-3B-interleave-grpo"

# Training hyperparameters
LEARNING_RATE = 5e-6
NUM_TRAIN_EPOCHS = 1
PER_DEVICE_TRAIN_BATCH_SIZE = 1

# IMPORTANT: generation_batch_size = batch * grad_accum * num_gpus
# must be divisible by num_generations
# With 1 GPU: 1 * 16 * 1 = 16, divisible by 16 ✓
GRADIENT_ACCUMULATION_STEPS = 16
NUM_GENERATIONS = 16  # Higher = better GRPO gradient estimates

# Generation
MAX_PROMPT_LENGTH = 2000
MAX_COMPLETION_LENGTH = 2500  # needs to fit ~1000 interleaved words

# Logging and saving
LOGGING_STEPS = 10
SAVE_STEPS = 50

# Sanity check config
LOG_SAMPLES_EVERY = 10  # Log samples every N steps
SAMPLE_LOG_FILE = "sample_outputs.log"

# ============================================================================
# SANITY CHECK CALLBACK
# ============================================================================

# Global storage for passing data from reward function to callback
_step_samples = {"data": []}


class SanityCheckCallback(TrainerCallback):
    """Log prompt + best/worst completions every N steps for debugging."""
    
    def __init__(self, log_every=10, log_file="sample_outputs.log"):
        self.log_every = log_every
        self.log_file = log_file
        
    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step % self.log_every != 0:
            return
        
        samples = _step_samples.get("data", [])
        if not samples:
            return
        
        with open(self.log_file, "a") as f:
            f.write(f"\n{'='*70}\n")
            f.write(f"STEP {state.global_step} | {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"{'='*70}\n\n")
            
            # Log first sample in detail
            item = samples[0]
            f.write(f"PROMPT (first 500 chars):\n{item['prompt'][:500]}...\n\n")
            f.write(f"EXPECTED (first 50 words):\n{item['expected']}\n\n")
            f.write(f"BEST (reward={item['best_reward']:.4f}):\n{item['best'][:500]}...\n\n")
            f.write(f"WORST (reward={item['worst_reward']:.4f}):\n{item['worst'][:500]}...\n\n")
            
            # Summary stats
            all_rewards = [s['best_reward'] for s in samples] + [s['worst_reward'] for s in samples]
            f.write(f"BATCH STATS: best_max={max(s['best_reward'] for s in samples):.4f}, ")
            f.write(f"worst_min={min(s['worst_reward'] for s in samples):.4f}, ")
            f.write(f"spread={max(all_rewards)-min(all_rewards):.4f}\n")
        
        # Clear for next step
        _step_samples["data"] = []


# ============================================================================
# REWARD FUNCTION WITH LOGGING
# ============================================================================

def interleave_reward_func(completions, expected, prompts=None, **kwargs):
    """
    Reward function that scores completions and captures samples for sanity logging.
    """
    rewards = []
    batch_data = []
    
    for i, (completion_group, expected_tokens) in enumerate(zip(completions, expected)):
        group_rewards = []
        group_texts = []
        
        for completion in completion_group:
            # Handle both dict and string completions
            if isinstance(completion, dict):
                text = completion.get("content", "")
            elif isinstance(completion, list) and len(completion) > 0:
                text = completion[0].get("content", "") if isinstance(completion[0], dict) else str(completion[0])
            else:
                text = str(completion)
            
            tokens = text.split()
            score = nw_align(tokens, expected_tokens)
            group_rewards.append(score)
            group_texts.append(text)
        
        rewards.extend(group_rewards)
        
        # Capture best/worst for logging
        if group_rewards and len(set(group_rewards)) > 1:  # Has variance
            best_idx = group_rewards.index(max(group_rewards))
            worst_idx = group_rewards.index(min(group_rewards))
            
            prompt_text = "[no prompt]"
            if prompts is not None and i < len(prompts):
                p = prompts[i]
                if isinstance(p, dict):
                    prompt_text = p.get("content", str(p))
                elif isinstance(p, list) and len(p) > 0:
                    prompt_text = p[0].get("content", str(p[0])) if isinstance(p[0], dict) else str(p[0])
                else:
                    prompt_text = str(p)
            
            batch_data.append({
                "prompt": prompt_text,
                "expected": " ".join(expected_tokens[:50]) + ("..." if len(expected_tokens) > 50 else ""),
                "best": group_texts[best_idx],
                "best_reward": group_rewards[best_idx],
                "worst": group_texts[worst_idx],
                "worst_reward": group_rewards[worst_idx],
            })
    
    # Store for callback to pick up
    _step_samples["data"] = batch_data
    
    return rewards


# ============================================================================
# CHECKPOINT RESUME
# ============================================================================

def get_latest_checkpoint(output_dir):
    """Find the latest checkpoint in output directory."""
    if not os.path.exists(output_dir):
        return None
    
    checkpoints = [d for d in os.listdir(output_dir) if d.startswith("checkpoint-")]
    if not checkpoints:
        return None
    
    # Sort by step number
    checkpoints.sort(key=lambda x: int(x.split("-")[1]))
    latest = os.path.join(output_dir, checkpoints[-1])
    return latest


# ============================================================================
# TRAINING
# ============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", action="store_true", help="Resume from latest checkpoint")
    parser.add_argument("--test", action="store_true", help="Run smoke test (1 step only)")
    parser.add_argument("--dataset", default=None, help="Path to JSONL dataset (if not provided, generates fresh)")
    args = parser.parse_args()
    
    print("=" * 50)
    print("Interleave GRPO Training")
    print("=" * 50)
    
    # Check for resume
    resume_checkpoint = None
    if args.resume:
        resume_checkpoint = get_latest_checkpoint(OUTPUT_DIR)
        if resume_checkpoint:
            print(f"\n>>> Will resume from: {resume_checkpoint}")
        else:
            print("\n>>> --resume specified but no checkpoint found. Starting fresh.")
    
    # Load dataset
    print("\n>>> Loading dataset...")
    if args.dataset:
        print(f"    Using: {args.dataset}")
    train_dataset, val_dataset, test_dataset = generate_dataset(dataset_path=args.dataset)
    print(f"    Train: {len(train_dataset)}")
    print(f"    Val: {len(val_dataset)}")
    print(f"    Test: {len(test_dataset)} (held out)")
    
    # Show sample prompt for sanity check
    print("\n>>> Sample from training set:")
    sample = train_dataset[0]
    prompt_content = sample['prompt'][0]['content'] if isinstance(sample['prompt'], list) else sample['prompt']
    print(f"    Prompt (first 200 chars): {prompt_content[:200]}...")
    print(f"    Expected tokens: {len(sample['expected'])} total, first 10: {sample['expected'][:10]}")
    
    # Load model and tokenizer
    print(f"\n>>> Loading model: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        device_map=None,
    ).to("cuda")
    
    print(f"    Parameters: {model.num_parameters():,}")
    
    # Configure training
    print("\n>>> Configuring GRPO...")
    
    max_steps = -1 if not args.test else 1
    
    training_args = GRPOConfig(
        output_dir=OUTPUT_DIR,
        run_name=RUN_NAME,
        
        # Optimizer - constant LR, let Adam adapt
        learning_rate=LEARNING_RATE,
        lr_scheduler_type='constant',
        max_grad_norm=1.0,
        
        # Batch sizes
        per_device_train_batch_size=PER_DEVICE_TRAIN_BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        
        # GRPO specific
        num_generations=NUM_GENERATIONS,
        max_prompt_length=MAX_PROMPT_LENGTH,
        max_completion_length=MAX_COMPLETION_LENGTH,
        
        # Training duration
        num_train_epochs=NUM_TRAIN_EPOCHS,
        max_steps=max_steps,
        
        # No mid-training eval (use evaluate.py before/after instead)
        eval_strategy="no",
        
        # Logging and saving
        logging_steps=LOGGING_STEPS,
        save_steps=SAVE_STEPS,
        save_total_limit=10,
        save_only_model=False,  # Include optimizer state - required for resume
        
        # Technical
        bf16=True,
        remove_unused_columns=False,  # Keep 'expected' column for reward
        report_to="wandb",
        log_on_each_node=False,
    )
    
    # Initialize sanity check callback
    sanity_callback = SanityCheckCallback(
        log_every=LOG_SAMPLES_EVERY,
        log_file=SAMPLE_LOG_FILE
    )
    
    # Create trainer
    print("\n>>> Initializing trainer...")
    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=[interleave_reward_func],
        args=training_args,
        train_dataset=train_dataset,
        callbacks=[sanity_callback],
    )
    
    # Print config summary
    print(f"\n>>> Config summary:")
    print(f"    max_prompt_length: {MAX_PROMPT_LENGTH}")
    print(f"    max_completion_length: {MAX_COMPLETION_LENGTH}")
    print(f"    num_generations: {NUM_GENERATIONS}")
    print(f"    Sample log: {SAMPLE_LOG_FILE} (every {LOG_SAMPLES_EVERY} steps)")
    
    # Train
    print("\n>>> Starting training...")
    print("=" * 50)
    
    if resume_checkpoint:
        print(f"Resuming from {resume_checkpoint}")
        trainer.train(resume_from_checkpoint=resume_checkpoint)
    else:
        trainer.train()
    
    print("\n>>> Training complete!")
    print(f"    Checkpoints saved to: {OUTPUT_DIR}")
    print(f"    Sample outputs logged to: {SAMPLE_LOG_FILE}")


if __name__ == "__main__":
    main()