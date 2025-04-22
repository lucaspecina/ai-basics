import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments # Removed Trainer, TrainerCallback
from datasets import load_dataset
from trl import DPOTrainer # Added DPOTrainer
import os 
import json # Added for saving progress
import argparse # Added for command-line arguments

# --- Argument Parsing ---
def parse_args():
    parser = argparse.ArgumentParser(description="Perform DPO training on a Llama model.") # Updated description
    parser.add_argument(
        "--model_id", 
        type=str, 
        # Default to the SFT model path eventually? Or keep base model? Let's keep base for now.
        default="meta-llama/Llama-3.2-1B-Instruct", 
        help="The model ID from Hugging Face to use for DPO." 
    )
    # Add argument for potentially loading a reference model separately if needed for DPO
    # parser.add_argument("--ref_model_id", type=str, default=None, help="Optional reference model ID for DPO.")
    parser.add_argument(
        "--data_file_name", 
        type=str, 
        default="dpo_data.json", # Changed default to .json
        help="The name of the JSON file in the data directory to use for DPO (must contain 'instruction', 'input', 'chosen', 'rejected')." # Updated help text
    )
    parser.add_argument(
        "--beta", 
        type=float, 
        default=0.1, 
        help="The beta parameter for DPO loss."
    )
    # Add other arguments here if needed in the future (e.g., max_length, max_prompt_length)
    return parser.parse_args()

args = parse_args()
# --- End Argument Parsing ---

# --- Global Config & Paths (adjust as needed) ---
ROOT_DIR = r"C:\\\\Users\\\\YT40432\\\\Desktop\\\\lp\\\\research\\\\lucaspecina\\\\ai-basics"
# Updated paths for DPO structure
DATA_PATH_prefix = os.path.join(ROOT_DIR, "ai-basics", "DPO", "llama", "data") 
# Removing PROGRESS_LOG_PATH for now, can add specific DPO monitoring later
# PROGRESS_LOG_PATH_prefix = os.path.join(ROOT_DIR, 'ai-basics', 'DPO', 'llama', 'training_progress_results') 
DATA_PATH = os.path.join(DATA_PATH_prefix, args.data_file_name) 
data_file_name_base = os.path.basename(DATA_PATH).replace('.json', '') # Use .json
print(f"Using data file: {data_file_name_base}")
# PROGRESS_LOG_PATH = os.path.join(PROGRESS_LOG_PATH_prefix, f"dpo_progress-{data_file_name_base}.json")
MODELS_DIR = os.path.join(ROOT_DIR, "models", "dpo-llama") # Updated model save directory

# Removed FIXED_QUESTIONS and related logic
# FIXED_QUESTIONS = [...]

# Determine device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Removed LOGGING_STEP_INTERVAL as we removed the custom callback
# LOGGING_STEP_INTERVAL = 10 

# --- Helper Functions & Classes ---

# Removed generate_responses function (can be added back if needed for evaluation)

# Removed apply_chat_template_map and tokenize_function_map

# Function to create the prompt column for DPO
def create_prompt_column(example):
    """
    Combines 'instruction' and 'input' into a single 'prompt' string.
    DPOTrainer will later apply the chat template.
    """
    # Simple combination. Add more complex formatting if needed.
    if example.get('input'):
        example['prompt'] = f"Instruction: {example['instruction']}\\nInput: {example['input']}"
    else:
        example['prompt'] = f"Instruction: {example['instruction']}"
    return example

# Removed preprocess_dpo_dataset placeholder

# Removed ProgressMonitorCallback class

# --- End Helper Functions & Classes ---

# --- Main Function ---
def main(args):
    model_id = args.model_id
    print(f"--- Using device: {DEVICE} ---")
    print(f"--- Loading base model for DPO: {model_id} ---")

    # Load model and tokenizer
    # For DPO, we might load the SFT model as the base
    # model_id_or_path = os.path.join(ROOT_DIR, "models", "sft-llama", "fine-tuned-model") # Example path
    # model = AutoModelForCausalLM.from_pretrained(model_id_or_path, ...)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float32, # Keep float32 for potential CPU/MPS usage
        device_map=DEVICE,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token # Ensure pad token is set

    # Load and prepare dataset for DPO
    print("--- Loading DPO dataset ---")
    # Load JSON data
    try:
        # Load the dataset from JSON file
        dataset = load_dataset("json", data_files=DATA_PATH, split="train")
        
        # Create the 'prompt' column
        dataset = dataset.map(create_prompt_column)

        # Select and rename columns for DPOTrainer: 'prompt', 'chosen', 'rejected'
        # Remove original columns to avoid issues with DPOTrainer's internal processing
        columns_to_keep = ["prompt", "chosen", "rejected"]
        columns_to_remove = [col for col in dataset.column_names if col not in columns_to_keep]
        dataset = dataset.remove_columns(columns_to_remove)
        
        print("--- Dataset preprocessed for DPO ---")
        print(f"Dataset columns: {dataset.column_names}")

        # Split dataset if necessary (DPOTrainer can take eval_dataset)
        # Check if split needs to be done after preprocessing
        if 'train' not in dataset.column_names and 'test' not in dataset.column_names: # This check might be incorrect after map/remove
             # Assuming the whole loaded dataset is for training if no split exists yet
             # Re-assess splitting logic if needed
             try:
                split_dataset = dataset.train_test_split(test_size=0.05) # Example split
                train_dataset = split_dataset["train"]
                eval_dataset = split_dataset["test"]
                print(f"--- Dataset split into train ({len(train_dataset)}) and test ({len(eval_dataset)}) sets ---")
             except Exception as split_error:
                 print(f"Could not split dataset automatically: {split_error}. Using entire dataset for training.")
                 train_dataset = dataset
                 eval_dataset = None

        else: # Handle cases where dataset might already have splits (less likely with load_dataset("json", split="train"))
            train_dataset = dataset # Assume loaded split="train" is the train set
            eval_dataset = None # DPOTrainer can handle None eval_dataset
            # Or load a separate eval dataset if available
            # eval_dataset = load_dataset("json", data_files=EVAL_DATA_PATH, split="train").map(create_prompt_column).remove_columns(...)

        print("--- DPO Dataset ready ---")
        print(f"Train dataset size: {len(train_dataset)}")
        if eval_dataset:
            print(f"Eval dataset size: {len(eval_dataset)}")

    except Exception as e:
        print(f"Error loading or processing dataset: {e}")
        print("Please ensure the JSON file exists at the specified path and has 'instruction', 'input' (optional), 'chosen', 'rejected' fields.")
        return # Exit if dataset loading fails

    # Removed SFT progress monitoring setup

    # Define Training Arguments for DPO
    # Note: DPOTrainer uses TrainingArguments, but some args might behave differently
    # or specific DPO args are passed directly to DPOTrainer constructor
    training_args = TrainingArguments(
        output_dir=os.path.join(MODELS_DIR, "results"),
        # eval_strategy="steps", # Evaluation strategy might differ for DPO
        # eval_steps=40, 
        logging_steps=10, # Log metrics more frequently perhaps
        save_steps=100, 
        per_device_train_batch_size=1, # DPO often requires smaller batches
        # per_device_eval_batch_size=1, 
        gradient_accumulation_steps=4, # Accumulate gradients to simulate larger batch size
        num_train_epochs=1, # DPO often requires fewer epochs
        fp16=(DEVICE.type == 'cuda'),
        report_to="none",
        log_level="info",
        learning_rate=5e-7, # DPO usually requires a lower learning rate than SFT
        max_grad_norm=1.0, # Gradient clipping
        save_total_limit=2,
        remove_unused_columns=False, # Important for DPOTrainer
        # lr_scheduler_type='cosine', # Optional: learning rate scheduler
        # warmup_steps=10,           # Optional: warmup steps
    )

    # Initialize DPOTrainer
    # We might need a reference model (model_ref) if not using the base model implicitly
    # model_ref=None means the trainer will create a copy of the model before training
    dpo_trainer = DPOTrainer(
        model,
        ref_model=None, # Let trainer handle creation of reference model
        args=training_args,
        beta=args.beta,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset, # Optional
        tokenizer=tokenizer,
        # max_prompt_length=128, # Example: Define max lengths if needed
        # max_length=256,        # Example: Max length for generated sequences
    )

    # Train the model using DPO
    print("--- Starting DPO training ---")
    dpo_trainer.train()
    print("--- DPO Training finished ---")

    # Save the final DPO model and tokenizer
    final_model_path = os.path.join(MODELS_DIR, "dpo-fine-tuned-model")
    print(f"--- Saving final DPO model to {final_model_path} ---")
    dpo_trainer.save_model(final_model_path) 
    # Make sure tokenizer is saved correctly
    if hasattr(dpo_trainer, 'tokenizer') and dpo_trainer.tokenizer is not None:
         dpo_trainer.tokenizer.save_pretrained(final_model_path)
    else:
         tokenizer.save_pretrained(final_model_path) # Fallback to original tokenizer
    print("--- DPO Model and tokenizer saved ---")
# --- End Main Function ---

# --- Script Entry Point ---
if __name__ == "__main__":
    cli_args = parse_args()
    main(cli_args)
# --- End Script Entry Point ---

