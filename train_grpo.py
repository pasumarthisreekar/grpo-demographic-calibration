import torch
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig
from trl import GRPOTrainer, GRPOConfig
from transformers import BitsAndBytesConfig
import torch

# 1. Import your exact reward function from the other file
from reward_function import distribution_reward_func

# 2. Configuration
# Point this to your local Llama-3-8B-Instruct directory, or the Hugging Face hub ID
MODEL_ID = "meta-llama/Meta-Llama-3-8B-Instruct" 

# 3. Load Tokenizer
print("Loading Tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# 4. Construct the Dataset
# TRL's GRPOTrainer is incredibly smart. Any extra columns you put in this dataset 
# (like 'ground_truth_distribution') are automatically passed into your reward 
# function's arguments. This makes your custom reward function work natively!
data = [
    {
        "prompt": "Answer the following survey question realistically: Do you favor or oppose the death penalty for persons convicted of murder?",
        "ground_truth_distribution": {"Favor": 0.60, "Oppose": 0.40}
    },
    {
        "prompt": "Answer the following survey question realistically: Should marijuana be legalized?",
        "ground_truth_distribution": {"Legalize": 0.70, "Keep illegal": 0.30}
    }
]
dataset = Dataset.from_list(data)

# 5. LoRA Configuration (Crucial for VRAM)
# We only train a tiny adapter on top of Llama, which keeps VRAM usage extremely low.
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    task_type="CAUSAL_LM",
)

# 6. GRPO Training Parameters
training_args = GRPOConfig(
    output_dir="./llama-grpo-outputs",
    learning_rate=5e-6,              # RL requires very low learning rates compared to standard fine-tuning
    num_train_epochs=1,
    per_device_train_batch_size=1,   # Number of prompts processed at once
    gradient_accumulation_steps=4,
    num_generations=8,               # The 'G' in GRPO. Generates 8 answers per prompt to calculate Advantage.
    max_completion_length=50,        # Keep short so Llama doesn't ramble during survey answers
    bf16=True,                       # Use bf16 if you have an Ampere/Hopper GPU for speed
    logging_steps=1,
    remove_unused_columns=False      # Prevents TRL from deleting your 'ground_truth_distribution' column
)

# 7. Initialize Trainer
print("Initializing GRPO Trainer...")
trainer = GRPOTrainer(
    model=MODEL_ID,
    reward_funcs=[distribution_reward_func],
    args=training_args,
    train_dataset=dataset,
    peft_config=lora_config
)

# 8. Start the Engine
print("🚀 Starting GRPO Training Loop...")
trainer.train()

# 9. Save the mathematically aligned model
trainer.save_model("./llama-grpo-final")
tokenizer.save_pretrained("./llama-grpo-final")
print("✅ Training complete. Model safely written to disk.")