import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from src.optimizer import SWAG
from src.utils import load_config
import os

def main():
    # Load configuration
    cfg = load_config()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Paths for the saved SWAG model and state
    swag_path = cfg['save_path'] + "_swag"
    swag_state_path = os.path.join(swag_path, "swag_state.pt")
    
    print(f"Loading model and tokenizer from: {swag_path}")
    tokenizer = GPT2Tokenizer.from_pretrained(swag_path)
    # The saved model in swag_path is the SWA Mean model
    model = GPT2LMHeadModel.from_pretrained(swag_path).to(device)
    model.eval()

    # Initialize SWAG tracker and load the collected moments (mean, sq_mean, cov_mat_sqrt)
    print(f"Loading SWAG state from: {swag_state_path}")
    swag_model = SWAG(model, max_num_models=cfg['swag']['max_num_models'])
    swag_model.load(swag_state_path)
    
    # Input prompt
    test_prompt = "The James Webb Space Telescope (JWST) successfully deployed its"
    inputs = tokenizer(test_prompt, return_tensors="pt").to(device)
    
    num_samples = 3
    print(f"\n--- Comparing {num_samples} samples from SWAG posterior ---")
    
    sample_data = []

    # Helper function to get model output
    def get_sample_output(name):
        with torch.no_grad():
            # 1. Get Logits for the next token
            outputs = model(**inputs)
            next_token_logits = outputs.logits[0, -1, :]
            top_logit_val, top_token_id = torch.max(next_token_logits, dim=-1)
            top_token = tokenizer.decode([top_token_id.item()])
            
            # 2. Generate text
            gen_ids = model.generate(
                **inputs, 
                max_new_tokens=15, 
                do_sample=False, 
                pad_token_id=tokenizer.eos_token_id
            )
            generated_text = tokenizer.decode(gen_ids[0], skip_special_tokens=True)
            
            return {
                'name': name,
                'logits': next_token_logits.cpu().clone(),
                'top_token': top_token.strip(),
                'top_logit': top_logit_val.item(),
                'text': generated_text.replace('\n', ' ')
            }

    # Evaluate SWA Mean (current state of model after loading)
    print("Evaluating SWA Mean model...")
    swag_model.get_mean_model()
    sample_data.append(get_sample_output("SWA Mean"))
    
    # Evaluate Samples
    for i in range(num_samples):
        print(f"Drawing Sample {i+1} (scale={cfg['swag']['scale']})...")
        swag_model.sample(scale=cfg['swag']['scale'])
        sample_data.append(get_sample_output(f"Sample {i+1}"))

    # Display Results
    print("\n" + "="*80)
    print(f"{'Model Variant':<15} | {'Top Token':<10} | {'Logit':<10} | {'Generated Text'}")
    print("-" * 80)
    for data in sample_data:
        print(f"{data['name']:<15} | {data['top_token']:<10} | {data['top_logit']:<10.4f} | {data['text']}")
    print("="*80)

    # Check for differences in logits
    print("\nNumerical Difference Analysis (Logit differences relative to SWA Mean):")
    mean_logits = sample_data[0]['logits']
    for i in range(1, len(sample_data)):
        sample_logits = sample_data[i]['logits']
        # Calculate Mean Squared Error between logit vectors
        mse = torch.mean((mean_logits - sample_logits)**2).item()
        # Calculate Max difference
        max_diff = torch.max(torch.abs(mean_logits - sample_logits)).item()
        print(f"SWA Mean vs {sample_data[i]['name']}: MSE={mse:.2e}, Max Diff={max_diff:.6f}")

    # Conclusion
    all_texts = [d['text'] for d in sample_data]
    if len(set(all_texts)) > 1:
        print("\nCONCLUSION: Sampled models are PRODUCING DIFFERENT text outputs. Posterior sampling is working.")
    else:
        print("\nCONCLUSION: Sampled models produced identical text. Check if logit differences are large enough to affect greedy decoding.")

if __name__ == "__main__":
    main()
