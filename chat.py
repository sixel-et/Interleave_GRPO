#!/usr/bin/env python3
"""
Simple interactive chat script for testing trained models.
Usage: python chat.py [model_path]
"""

import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

BASE_MODEL = "meta-llama/Llama-3.2-3B-Instruct"

def main():
    parser = argparse.ArgumentParser(description="Chat with a trained model")
    parser.add_argument("model", nargs="?", default="./outputs/checkpoint-3900",
                        help="Path to model checkpoint (default: ./outputs/checkpoint-3900)")
    parser.add_argument("--base-tokenizer", default=BASE_MODEL,
                        help=f"Base model for tokenizer (default: {BASE_MODEL})")
    parser.add_argument("--max-tokens", type=int, default=256,
                        help="Max new tokens to generate (default: 256)")
    parser.add_argument("--temperature", type=float, default=0.7,
                        help="Sampling temperature (default: 0.7)")
    args = parser.parse_args()

    print(f"Loading tokenizer from {args.base_tokenizer}...")
    tokenizer = AutoTokenizer.from_pretrained(args.base_tokenizer)
    
    print(f"Loading model from {args.model}...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    print(f"Model loaded on {model.device}\n")
    
    print("=" * 50)
    print("Interactive Chat (type 'quit' to exit)")
    print("=" * 50)
    
    conversation = []
    
    while True:
        try:
            user_input = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break
            
        if user_input.lower() in ('quit', 'exit', 'q'):
            print("Goodbye!")
            break
        
        if user_input.lower() == 'clear':
            conversation = []
            print("[Conversation cleared]")
            continue
            
        if not user_input:
            continue
        
        conversation.append({"role": "user", "content": user_input})
        
        inputs = tokenizer.apply_chat_template(
            conversation, 
            return_tensors="pt",
            add_generation_prompt=True
        ).to(model.device)
        
        with torch.no_grad():
            outputs = model.generate(
                inputs,
                max_new_tokens=args.max_tokens,
                temperature=args.temperature,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id
            )
        
        response = tokenizer.decode(
            outputs[0][inputs.shape[1]:], 
            skip_special_tokens=True
        )
        
        conversation.append({"role": "assistant", "content": response})
        print(f"\nModel: {response}")


if __name__ == "__main__":
    main()
