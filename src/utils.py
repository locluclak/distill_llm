import torch
import os
import yaml
from transformers import TrainerCallback
from transformers import DataCollatorForLanguageModeling
from torch.utils.data import DataLoader
import math


def load_config(config_path="config.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)
def evaluate_ppl(model, eval_dataset, tokenizer, device="cuda", name="Model"):
    """Replicates the notebook's Perplexity evaluation logic."""
    model.eval()
    total_loss = 0
    
    # Use a DataLoader for efficiency
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    eval_dataloader = DataLoader(eval_dataset, batch_size=4, collate_fn=data_collator)

    print(f"--- Evaluating {name} ---")
    
    with torch.no_grad():
        for batch in eval_dataloader:
            # Move batch to device
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            
            outputs = model(input_ids, labels=labels)
            total_loss += outputs.loss.item() * input_ids.size(0)

    avg_loss = total_loss / len(eval_dataset)
    perplexity = math.exp(avg_loss)
    
    print(f"{name} Perplexity: {perplexity:.2f}")
    return perplexity

class WeightCollectionCallback(TrainerCallback):
    """Saves weight samples from the student model during training to a local directory."""
    def __init__(self, collection_steps, save_dir="./samples"):
        self.collection_steps = collection_steps
        self.save_dir = save_dir
        os.makedirs(self.save_dir, exist_ok=True)

    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step in self.collection_steps:
            model = kwargs['model']
            # Move to CPU to save VRAM
            sample = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            
            sample_path = os.path.join(self.save_dir, f"sample_step_{state.global_step}.pt")
            torch.save(sample, sample_path)
            
            print(f"\n[SGLD] Sample collected and saved to {sample_path}")