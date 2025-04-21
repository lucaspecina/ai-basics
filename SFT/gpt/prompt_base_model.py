#!/usr/bin/env python3
"""
Interactive CLI script to prompt the GPT base model.
"""
import argparse
import torch
import tiktoken
from gpt.gpt2_base_download import download_and_load_gpt2
from gpt.gpt_model import GPTModel, load_weights_into_gpt, text_to_token_ids, token_ids_to_text, generate

def main():
    parser = argparse.ArgumentParser(description="Interactive GPT2 base model prompt")
    parser.add_argument("--model_size", type=str, default="124M",
                        choices=["124M", "355M", "774M", "1558M"],
                        help="Size of GPT2 model to use")
    parser.add_argument("--models_dir", type=str, default="gpt2",
                        help="Directory to store or load pretrained GPT2 models")
    parser.add_argument("--max_new_tokens", type=int, default=50,
                        help="Number of tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.0,
                        help="Sampling temperature (>0 for sampling)")
    parser.add_argument("--top_k", type=int, default=None,
                        help="Top-k sampling; keep only top k tokens")
    parser.add_argument("--device", type=str, default=None,
                        help="Torch device, e.g. 'cuda' or 'cpu'")
    args = parser.parse_args()

    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))

    print(f"Loading GPT2 {args.model_size} model on {device}...")
    settings, params = download_and_load_gpt2(model_size=args.model_size, models_dir=args.models_dir)
    base_cfg = {
        "vocab_size": settings["n_vocab"],
        "context_length": settings["n_ctx"],
        "emb_dim": settings["n_embd"],
        "n_layers": settings["n_layer"],
        "n_heads": settings["n_head"],
        "drop_rate": settings.get("resid_pdrop", 0.0),
        "qkv_bias": True
    }
    model = GPTModel(base_cfg).to(device)
    load_weights_into_gpt(model, params)
    model.eval()

    tokenizer = tiktoken.get_encoding("gpt2")

    print("Ready to generate. Type your prompt (or 'exit' to quit).")
    while True:
        try:
            prompt = input(">>> ")
        except (EOFError, KeyboardInterrupt):
            print("\nExiting.")
            break
        if not prompt or prompt.strip().lower() in ("exit", "quit"):
            print("Goodbye!")
            break
        idx = text_to_token_ids(prompt, tokenizer).to(device)
        with torch.no_grad():
            output_idx = generate(
                model=model,
                idx=idx,
                max_new_tokens=args.max_new_tokens,
                context_size=base_cfg["context_length"],
                temperature=args.temperature,
                top_k=args.top_k
            )
        full_text = token_ids_to_text(output_idx, tokenizer)
        # Remove prompt from generated text
        gen_text = full_text[len(prompt):]
        print(gen_text)

if __name__ == "__main__":
    main()
