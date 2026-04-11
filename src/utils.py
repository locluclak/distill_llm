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
    """Callback for SWAG to collect model weights at specific intervals (Training Phase)."""
    def __init__(self, swag_model, start_step=0, interval=100, swa_lr=None):
        """
        Args:
            swag_model (SWAG): The SWAG tracker instance.
            start_step (int): Step to start collecting weights (after burn-in).
            interval (int): Collect weights every N steps.
            swa_lr (float, optional): Learning rate to use during SWAG phase. 
                                     If None, will use the LR at start_step.
        """
        self.swag_model = swag_model
        self.start_step = start_step
        self.interval = interval
        self.swa_lr = swa_lr
        self._swa_lr_set = False

    def on_step_begin(self, args, state, control, **kwargs):
        """Ensure learning rate is constant during SWAG phase."""
        if state.global_step >= self.start_step:
            optimizer = kwargs.get("optimizer")
            scheduler = kwargs.get("lr_scheduler")

            if optimizer:
                # If swa_lr is not set, capture current LR at the start of SWAG phase
                if self.swa_lr is None:
                    self.swa_lr = optimizer.param_groups[0]['lr']
                    print(f"\n[SWAG] Step {state.global_step}: Capturing SWAG LR: {self.swa_lr}")

                # Force constant LR for all parameter groups in the optimizer
                for param_group in optimizer.param_groups:
                    param_group['lr'] = self.swa_lr

            if scheduler and self.swa_lr is not None:
                # Override scheduler's base_lrs and last_lr to stop it from decaying in logs/steps
                if hasattr(scheduler, 'base_lrs'):
                    scheduler.base_lrs = [self.swa_lr] * len(optimizer.param_groups)
                if hasattr(scheduler, '_last_lr'):
                    scheduler._last_lr = [self.swa_lr] * len(optimizer.param_groups)

    def on_step_end(self, args, state, control, **kwargs):
        # Override LR again in case the scheduler updated it at the end of the step
        if state.global_step >= self.start_step:
            optimizer = kwargs.get("optimizer")
            scheduler = kwargs.get("lr_scheduler")

            if optimizer and self.swa_lr is not None:
                for param_group in optimizer.param_groups:
                    param_group['lr'] = self.swa_lr

            if scheduler and self.swa_lr is not None:
                if hasattr(scheduler, '_last_lr'):
                    scheduler._last_lr = [self.swa_lr] * len(optimizer.param_groups)

            # Collect model at intervals
            if (state.global_step - self.start_step) % self.interval == 0:
                self.swag_model.collect_model()
                print(f"\n[SWAG] Step {state.global_step}: Model weights collected. LR forced to: {self.swa_lr}")