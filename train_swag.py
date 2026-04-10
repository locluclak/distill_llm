import os
import torch
from transformers import (
    GPT2LMHeadModel, 
    GPT2Tokenizer, 
    TrainingArguments, 
    DataCollatorForLanguageModeling, 
    BitsAndBytesConfig
)
from datasets import load_dataset
from peft import PeftModel
from src.trainer import SWAGDistillationTrainer
from src.optimizer import SWAG
from src.utils import load_config, SWAGCallback, evaluate_ppl

def main():
    # Load configuration
    cfg = load_config()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 1. Setup Tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained(cfg['base_model_name'])
    tokenizer.pad_token = tokenizer.eos_token
    
    # 2. Data Preparation
    dataset = load_dataset(cfg['dataset']['name'], cfg['dataset']['subset'])
    
    def tokenize_fn(examples):
        return tokenizer(examples["text"], truncation=True, max_length=cfg['dataset']['max_length'])

    tokenized_datasets = dataset.map(tokenize_fn, batched=True, remove_columns=["text"])
    train_data = tokenized_datasets["train"].filter(lambda x: len(x["input_ids"]) > 0)
    eval_data = tokenized_datasets["test"].select(range(min(100, len(tokenized_datasets["test"]))))

    # 3. Load Teacher (8-bit Quantized)
    quant_config = BitsAndBytesConfig(load_in_8bit=True)
    base_teacher = GPT2LMHeadModel.from_pretrained(
        cfg['base_model_name'], 
        quantization_config=quant_config, 
        device_map="auto"
    )
    teacher_model = PeftModel.from_pretrained(base_teacher, cfg['teacher_model_path'])
    teacher_model.eval()

    # 4. Load Student
    student_model = GPT2LMHeadModel.from_pretrained(cfg['student_model_name']).to("cuda")

    # 5. Prepare SWAG
    # This tracker ONLY monitors training progress and maintains mean/variance buffers.
    swag_model = SWAG(student_model, max_num_models=cfg['swag']['max_num_models'])
    
    # Calculate total steps for automatic SWAG start (75% of training)
    num_train_epochs = cfg['training']['epochs']
    batch_size = cfg['training']['batch_size']
    grad_accum = cfg['training']['gradient_accumulation_steps']
    # Number of optimization steps per epoch
    steps_per_epoch = len(train_data) // (batch_size * grad_accum)
    total_steps = steps_per_epoch * num_train_epochs
    
    # Use 75% as default if not overridden or if using the 75% rule
    swag_start_step = int(total_steps * 0.75)
    
    print(f"\n[SWAG Config] Total expected steps: {total_steps}")
    print(f"[SWAG Config] SWAG collection will start at step: {swag_start_step} (75% mark)")

    callbacks = []
    if cfg['swag']['enabled']:
        swag_callback = SWAGCallback(
            swag_model=swag_model,
            start_step=swag_start_step,
            interval=cfg['swag']['interval']
        )
        callbacks.append(swag_callback)

    # 6. Pre-Distillation Evaluation
    print("\n[PRE-TRAIN] Running initial evaluation...")
    teacher_ppl = evaluate_ppl(teacher_model, eval_data, tokenizer, device, "Teacher")
    student_pre_ppl = evaluate_ppl(student_model, eval_data, tokenizer, device, "Student (Base)")

    # 7. Distillation Setup
    # - Using 'logging_steps' to show loss during training
    # - Using standard optimizer (default) as SWAG is used on top of it.
    training_args = TrainingArguments(
        output_dir="./results_swag",
        per_device_train_batch_size=cfg['training']['batch_size'],
        gradient_accumulation_steps=cfg['training']['gradient_accumulation_steps'],
        num_train_epochs=cfg['training']['epochs'],
        learning_rate=float(cfg['training']['learning_rate']),
        fp16=cfg['training']['fp16'],
        logging_steps=10,  # Show log loss every 10 steps
        # eval_strategy="steps",
        # eval_steps=100,
        save_strategy="no",
        report_to="none"
    )

    trainer = SWAGDistillationTrainer(
        model=student_model,
        teacher_model=teacher_model,
        temperature=cfg['training']['temperature'],
        alpha=cfg['training']['alpha'],
        args=training_args,
        train_dataset=train_data,
        eval_dataset=eval_data,
        callbacks=callbacks,
        data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
    )

    # 8. Execution - PHASE 1: Moment Collection
    print("Starting SWAG Distillation Phase 1 (Training & Moment Collection)...")
    trainer.train()

    # 9. Execution - PHASE 2: Bayesian Model Averaging (BMA) & Sampling
    print("\nStarting Phase 2 (Inference Phase with SWAG)...")
    student_model.eval()
    
    # 9.1 Evaluate Standard Student (last weights from SGD/Adam)
    print("\n[Evaluation] Evaluating model with last training weights...")
    student_last_ppl = evaluate_ppl(student_model, eval_data, tokenizer, device, "Student (Last weights)")

    # 9.2 Evaluate SWA Mean (Best point estimate)
    print("\n[Evaluation] Evaluating SWA mean weights...")
    swag_model.get_mean_model()
    student_swa_ppl = evaluate_ppl(student_model, eval_data, tokenizer, device, "Student (SWA Mean)")

    # 9.3 Perform Bayesian Model Averaging (BMA) via sampling
    # Typically, we take multiple samples and ensemble their predictions. 
    # For perplexity, we can average the loss across samples.
    num_samples = 5
    sample_ppls = []
    
    # Prepare for output difference check
    test_prompt = "The future of artificial intelligence will be"
    test_input = tokenizer(test_prompt, return_tensors="pt").to(device)
    sample_outputs = []

    print(f"\n[Evaluation] Drawing {num_samples} samples from posterior for BMA...")
    for i in range(num_samples):
        swag_model.sample(scale=cfg['swag']['scale'])
        
        # Check output for this sample
        with torch.no_grad():
            outputs = student_model(**test_input)
            # Get logits for the last token to compare differences
            last_token_logits = outputs.logits[0, -1, :]
            top_token_id = torch.argmax(last_token_logits).item()
            top_token = tokenizer.decode([top_token_id])
            top_logit_val = last_token_logits[top_token_id].item()
            sample_outputs.append((top_token, top_logit_val))
        
        ppl = evaluate_ppl(student_model, eval_data, tokenizer, device, f"Student (Sample {i+1})")
        sample_ppls.append(ppl)
        print(f"\t Sample {i+1} Perplexity: {ppl:.2f} | Top prediction: '{top_token.strip()}' (logit: {top_logit_val:.4f})")

    # Check if outputs are different
    unique_tokens = set([out[0] for out in sample_outputs])
    logit_values = [out[1] for out in sample_outputs]
    logit_diff = max(logit_values) - min(logit_values)
    
    print(f"\n" + "-"*30)
    print(f"Diversity Check (Prompt: '{test_prompt}'):")
    print(f"Unique top-1 tokens: {len(unique_tokens)}")
    print(f"Max logit difference: {logit_diff:.6f}")
    
    if len(unique_tokens) > 1 or logit_diff > 1e-4:
        print("RESULT: Sampled models are PRODUCING DIFFERENT outputs. Posterior sampling is effective.")
    else:
        print("RESULT: Sampled models are producing identical outputs. Scaling or collection might be insufficient.")
    print("-"*30)

    avg_sample_ppl = sum(sample_ppls) / len(sample_ppls)

    # Final Summary Table
    print(f"\n" + "="*50)
    print(f"   FINAL RESULTS SUMMARY (SWAG)")
    print(f"="*50)
    print(f"Teacher PPL:           {teacher_ppl:.2f}")
    print(f"Student (Before):      {student_pre_ppl:.2f}")
    print(f"Student (Last Weights): {student_last_ppl:.2f}")
    print(f"Student (SWA Mean):    {student_swa_ppl:.2f}")
    print(f"Average Sample PPL:    {avg_sample_ppl:.2f}")
    print(f"="*50)

    # 10. Save results
    # Save the SWA mean model as the final distilled model
    swag_model.get_mean_model()
    student_model.save_pretrained(cfg['save_path'] + "_swag")
    tokenizer.save_pretrained(cfg['save_path'] + "_swag")
    
    # Save SWAG state for future sampling
    swag_model.save(os.path.join(cfg['save_path'] + "_swag", "swag_state.pt"))
    print(f"Model and SWAG state saved to {cfg['save_path']}_swag")

if __name__ == "__main__":
    main()
