import torch
import argparse
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from mingpt.model import GPT
from mingpt.utils import set_seed
from mingpt.bpe import BPETokenizer
set_seed(3407)

# Default settings
DEFAULT_MODEL_TYPE = 'gpt2' # Changed default to smaller model for quicker testing
DEFAULT_DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
DEFAULT_PROMPT = 'Lucas Pecina, the'
DEFAULT_NUM_SAMPLES = 3
DEFAULT_STEPS = 20

def setup_model(model_type, use_mingpt, device):
    if use_mingpt:
        model = GPT.from_pretrained(model_type)
    else:
        model = GPT2LMHeadModel.from_pretrained(model_type)
        model.config.pad_token_id = model.config.eos_token_id # suppress a warning

    # ship model to device and set to eval mode
    model.to(device)
    model.eval()
    print(f"Using {'minGPT' if use_mingpt else 'HuggingFace Transformers'} model: {model_type} on {device}")
    return model


def generate(model, tokenizer, prompt='', num_samples=10, steps=20, do_sample=True, device='cpu', use_mingpt=True):
    # tokenize the input prompt into integer input sequence
    if use_mingpt:
        if prompt == '':
            # to create unconditional samples...
            # manually create a tensor with only the special <|endoftext|> token
            # similar to what openai's code does here https://github.com/openai/gpt-2/blob/master/src/generate_unconditional_samples.py
            x = torch.tensor([[tokenizer.encoder.encoder['<|endoftext|>']]], dtype=torch.long)
        else:
            x = tokenizer(prompt).to(device)
    else:
        if prompt == '':
            # to create unconditional samples...
            # huggingface/transformers tokenizer special cases these strings
            prompt = '<|endoftext|>'
        encoded_input = tokenizer(prompt, return_tensors='pt').to(device)
        x = encoded_input['input_ids']
    
    # we'll process all desired num_samples in a batch, so expand out the batch dim
    x = x.expand(num_samples, -1)

    # forward the model `steps` times to get samples, in a batch
    # Use torch.no_grad() for inference
    with torch.no_grad():
        y = model.generate(x, max_new_tokens=steps, do_sample=do_sample, top_k=40)

    print(f"Prompt: \"{prompt}\"")
    for i in range(num_samples):
        out = tokenizer.decode(y[i].cpu().squeeze())
        print('-'*80)
        print(out)

def main():
    parser = argparse.ArgumentParser(description="Generate text using minGPT or HuggingFace GPT-2.")
    parser.add_argument('--use_mingpt', action='store_true', default=True, help='Use minGPT model (default: True)')
    parser.add_argument('--use_hf', action='store_false', dest='use_mingpt', help='Use HuggingFace Transformers model')
    parser.add_argument('--model_type', type=str, default=DEFAULT_MODEL_TYPE, help=f'GPT model type (e.g., gpt2, gpt2-medium, gpt2-large, gpt2-xl). Default: {DEFAULT_MODEL_TYPE}')
    parser.add_argument('--device', type=str, default=DEFAULT_DEVICE, help=f'Device to use (cuda or cpu). Default: {DEFAULT_DEVICE}')
    parser.add_argument('--prompt', type=str, default=DEFAULT_PROMPT, help=f'Input prompt. Default: "{DEFAULT_PROMPT}"')
    parser.add_argument('--num_samples', type=int, default=DEFAULT_NUM_SAMPLES, help=f'Number of samples to generate. Default: {DEFAULT_NUM_SAMPLES}')
    parser.add_argument('--steps', type=int, default=DEFAULT_STEPS, help=f'Number of steps (tokens) to generate. Default: {DEFAULT_STEPS}')
    parser.add_argument('--seed', type=int, default=3407, help='Random seed. Default: 3407')
    parser.add_argument('--no_sample', action='store_false', dest='do_sample', help='Use greedy decoding instead of sampling.')


    args = parser.parse_args()
    set_seed(args.seed)

    model = setup_model(args.model_type, args.use_mingpt, args.device)

    # Setup tokenizer based on whether using mingpt or HF
    if args.use_mingpt:
        tokenizer = BPETokenizer() # mingpt uses its own BPE tokenizer
    else:
        tokenizer = GPT2Tokenizer.from_pretrained(args.model_type) # HF uses its standard tokenizer

    generate(
        model=model,
        tokenizer=tokenizer,
        prompt=args.prompt,
        num_samples=args.num_samples,
        steps=args.steps,
        do_sample=args.do_sample,
        device=args.device,
        use_mingpt=args.use_mingpt
    )

if __name__ == "__main__":
    main()