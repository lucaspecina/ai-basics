import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments, TrainerCallback
from datasets import load_dataset
import os 
import json # Added for saving progress
import argparse # Added for command-line arguments
from functools import partial # Added import

# --- Argument Parsing ---
def parse_args():
    parser = argparse.ArgumentParser(description="Finetune a Llama model.")
    parser.add_argument(
        "--model_id", 
        type=str, 
        default="meta-llama/Llama-3.2-1B-Instruct", 
        help="The model ID from Hugging Face to finetune."
    )
    # Add other arguments here if needed in the future
    return parser.parse_args()

args = parse_args()
# --- End Argument Parsing ---

# --- Global Config & Paths (adjust as needed) ---
ROOT_DIR = r"C:\\Users\\YT40432\\Desktop\\lp\\research\\lucaspecina\\ai-basics"
MODELS_DIR = os.path.join(ROOT_DIR, "models\\sft-llama")
PROGRESS_LOG_PATH = os.path.join(MODELS_DIR, "training_progress.json")
DATA_PATH = "SFT/llama/data/sarcasm_shorter.csv"

# Fixed questions to monitor progress
FIXED_QUESTIONS = [
    "Explain the concept of supervised learning in simple terms.",
    "What is the capital of France?",
    "Write a short poem about a cat.",
    "Summarize the main idea of the theory of relativity.",
]

# Determine device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LOGGING_STEP_INTERVAL = 5 # Log responses every 20 steps

# --- Helper Functions & Classes ---
def generate_responses(model, tokenizer, questions, device):
    model.eval() # Set model to evaluation mode
    responses = {}
    with torch.no_grad():
        for q in questions:
            messages = [{"role": "user", "content": q}]
            prompt = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            inputs = tokenizer(prompt, return_tensors="pt").to(device)
            
            # Generate response
            outputs = model.generate(
                **inputs,
                max_new_tokens=100, # Limit response length
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id,
                do_sample=True, # Enable sampling for more varied responses
                temperature=0.7,
                top_p=0.9
            )
            response_text = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
            responses[q] = response_text
            print(f"Step/Initial - Q: {q} -> A: {response_text[:50]}...") # Print progress
    model.train() # Set model back to training mode
    return responses

# Define a function to apply the chat template
def apply_chat_template_map(example, tokenizer):
    messages = [
        {"role": "user", "content": example['question']},
        {"role": "assistant", "content": example['answer']}
    ]
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    return {"prompt": prompt}

# Tokenize the data
def tokenize_function_map(example, tokenizer):
    tokens = tokenizer(example['prompt'], padding="max_length", truncation=True, max_length=128)
    # Set padding token labels to -100 to ignore them in loss calculation
    tokens['labels'] = [
        -100 if token == tokenizer.pad_token_id else token for token in tokens['input_ids']
    ]
    return tokens

# Custom Callback for monitoring progress
class ProgressMonitorCallback(TrainerCallback):
    """Callback to generate and log model responses at specific step intervals."""
    def __init__(self, questions, log_path, device, logging_interval, tokenizer):
        self.questions = questions
        self.log_path = log_path
        self.device = device
        self.logging_interval = logging_interval
        self.progress_log = {}
        self.tokenizer = tokenizer

    def set_initial_log(self, log):
        self.progress_log = log

    def on_step_end(self, args, state, control, model=None, tokenizer=None, **kwargs):
        """Event called at the end of a training step."""
        if state.global_step > 0 and state.global_step % self.logging_interval == 0:
            
            current_model = model
            current_tokenizer = self.tokenizer

            if current_model is not None and current_tokenizer is not None:
                print(f"--- Generating responses at step {state.global_step} using model: {type(current_model)}, tokenizer: {type(current_tokenizer)} ---") 
                step_responses = generate_responses(current_model, current_tokenizer, self.questions, self.device)
                self.progress_log[state.global_step] = step_responses

                # Save progress incrementally
                try:
                    os.makedirs(os.path.dirname(self.log_path), exist_ok=True)
                    with open(self.log_path, 'w') as f:
                        json.dump(self.progress_log, f, indent=4)
                    print(f"--- Progress log saved to {self.log_path} ---")
                except Exception as e:
                    print(f"Error saving progress log at step {state.global_step}: {e}")
            else:
                 print(f"--- ProgressMonitorCallback: ERROR - Could not access model or tokenizer at step {state.global_step}! Model type: {type(current_model)}, Tokenizer type: {type(current_tokenizer)} ---") 

# --- End Helper Functions & Classes ---

# --- Main Function ---
def main(args):
    model_id = args.model_id
    print(f"--- Using device: {DEVICE} ---")
    print(f"--- Loading base model: {model_id} ---")

    # Load model and tokenizer
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float32,
        device_map=DEVICE,
         ) # Must be float32 for MacBooks!
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token

    # Load and prepare dataset
    print("--- Loading and preparing dataset ---")
    dataset = load_dataset("csv", data_files=DATA_PATH, split="train")

    # Use partial to pass tokenizer to map functions
    apply_chat_template_with_tokenizer = partial(apply_chat_template_map, tokenizer=tokenizer)
    tokenize_function_with_tokenizer = partial(tokenize_function_map, tokenizer=tokenizer)

    new_dataset = dataset.map(apply_chat_template_with_tokenizer)
    new_dataset = new_dataset.train_test_split(test_size=0.05) # Let's keep 5% of the data for testing
    tokenized_dataset = new_dataset.map(tokenize_function_with_tokenizer, batched=True) # Added batched=True for efficiency
    tokenized_dataset = tokenized_dataset.remove_columns(['question', 'answer', 'prompt'])
    print("--- Dataset prepared ---")

    # Progress Monitoring Setup
    progress_monitor = ProgressMonitorCallback(FIXED_QUESTIONS, PROGRESS_LOG_PATH, DEVICE, LOGGING_STEP_INTERVAL, tokenizer)
    print("--- Generating initial responses ---")
    initial_responses = generate_responses(model, tokenizer, FIXED_QUESTIONS, DEVICE)
    initial_log = {0: initial_responses} # Step 0 for base model
    progress_monitor.set_initial_log(initial_log)

    # Save initial log immediately
    try:
        os.makedirs(os.path.dirname(PROGRESS_LOG_PATH), exist_ok=True)
        with open(PROGRESS_LOG_PATH, 'w') as f:
            json.dump(initial_log, f, indent=4)
        print(f"--- Initial progress log saved to {PROGRESS_LOG_PATH} ---")
    except Exception as e:
        print(f"Error saving initial progress log: {e}")

    # Define Training Arguments
    model.train()
    training_args = TrainingArguments(
        output_dir=os.path.join(MODELS_DIR, "results"),
        eval_strategy="steps",
        eval_steps=40, # Keep eval frequency or adjust if needed
        logging_steps=40, # Keep metric logging frequency or adjust if needed
        save_steps=100, # Save checkpoints every 100 steps
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        num_train_epochs=2,
        fp16=(DEVICE.type == 'cuda'),
        report_to="none",
        log_level="info",
        learning_rate=1e-5,
        max_grad_norm=2,
        save_total_limit=3
    )

    # Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["test"],
        tokenizer=tokenizer,
        callbacks=[progress_monitor]
    )

    # Train the model
    print("--- Starting training ---")
    trainer.train()
    print("--- Training finished ---")

    # Save the final model and tokenizer
    final_model_path = os.path.join(MODELS_DIR, "fine-tuned-model")
    print(f"--- Saving final model to {final_model_path} ---")
    trainer.save_model(final_model_path)
    tokenizer.save_pretrained(final_model_path)
    print("--- Model and tokenizer saved ---")
# --- End Main Function ---

# --- Script Entry Point ---
if __name__ == "__main__":
    cli_args = parse_args()
    main(cli_args)
# --- End Script Entry Point ---

