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
    # Note: We skip the DataCollator for LM because evaluate_perplexity 
    # used raw input_ids as labels without special padding masks (-100).
    from torch.utils.data import DataLoader

    # Batch size 1 is required to match the "per-entry" average of evaluate_perplexity
    # If you use a higher batch size, you'd be averaging the average of batches, 
    # which is mathematically different.
    eval_dataloader = DataLoader(eval_dataset, batch_size=1)

    print(f"--- Evaluating {name} (Replicated Logic) ---")
    
    with torch.no_grad():
        for batch in eval_dataloader:
            # Replicating entry["input_ids"] logic
            # input_ids = batch["input_ids"].to(device)
            input_ids = torch.tensor(batch["input_ids"]).to(device)
            # Skip empty sequences as seen in your second function
            if input_ids.dim() == 1:
                input_ids = input_ids.unsqueeze(0)
            if input_ids.shape[1] == 0: 
                continue
            
            # In evaluate_perplexity, labels = input_ids
            outputs = model(input_ids, labels=input_ids)
            
            # Summing raw loss items (unweighted by tokens or batch size)
            total_loss += outputs.loss.item()

    # Averaging by the number of entries
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

class SWAGCallback(TrainerCallback):
    """Callback for SWAG to collect model weights at specific intervals."""
    def __init__(self, swag_model, collection_steps=None, collect_every_n_steps=None):
        """
        Args:
            swag_model (SWAG): The SWAG tracker instance.
            collection_steps (list, optional): Specific global steps to collect weights.
            collect_every_n_steps (int, optional): Collect weights every N steps.
        """
        self.swag_model = swag_model
        self.collection_steps = collection_steps or []
        self.collect_every_n_steps = collect_every_n_steps

    def on_step_end(self, args, state, control, **kwargs):
        should_collect = False
        if state.global_step in self.collection_steps:
            should_collect = True
        if self.collect_every_n_steps and state.global_step % self.collect_every_n_steps == 0:
            should_collect = True
        
        if should_collect:
            self.swag_model.collect_model()
            print(f"\n[SWAG] Model collected at step {state.global_step}. Total models: {self.swag_model.n_models}")