"""
interleave_grpo.py

GRPO training for the interleaving task.
Based on Will Brown's GSM8K demo.

Usage:
    python interleave_grpo.py
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import GRPOConfig, GRPOTrainer

from dataset_generator import generate_dataset
from reward import interleave_reward_func

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
MAX_PROMPT_LENGTH = 512
MAX_COMPLETION_LENGTH = 256  # our outputs are ~20 words but leave room

# Logging and saving
LOGGING_STEPS = 10
SAVE_STEPS = 50
EVAL_STEPS = 50

# ============================================================================
# TRAINING
# ============================================================================

def main():
    print("=" * 50)
    print("Interleave GRPO Training")
    print("=" * 50)
    
    # Load dataset
    print("\n>>> Loading dataset...")
    train_dataset, val_dataset, test_dataset = generate_dataset()
    print(f"    Train: {len(train_dataset)}")
    print(f"    Val: {len(val_dataset)}")
    print(f"    Test: {len(test_dataset)} (held out)")
    
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
    training_args = GRPOConfig(
        output_dir=OUTPUT_DIR,
        run_name=RUN_NAME,
        
        # Optimizer - constant LR, let Adam adapt
        learning_rate=LEARNING_RATE,
        lr_scheduler_type='constant',
        max_grad_norm=1.0,
        
        # Batch sizes
        per_device_train_batch_size=PER_DEVICE_TRAIN_BATCH_SIZE,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        
        # GRPO specific
        num_generations=NUM_GENERATIONS,
        num_generations_eval=1,  # Fewer generations during eval to save memory
        max_prompt_length=MAX_PROMPT_LENGTH,
        max_completion_length=MAX_COMPLETION_LENGTH,
        
        # Training duration
        num_train_epochs=NUM_TRAIN_EPOCHS,
        
        # Evaluation during training
        eval_strategy="steps",
        eval_steps=EVAL_STEPS,
        
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
    
    # Create trainer
    print("\n>>> Initializing trainer...")
    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=[interleave_reward_func],
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
    )
    
    # Train
    print("\n>>> Starting training...")
    print("=" * 50)
    trainer.train()
    
    print("\n>>> Training complete!")
    print(f"    Checkpoints saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()