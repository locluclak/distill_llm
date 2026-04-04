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
    swag_model = SWAG(student_model, max_num_models=cfg['swag']['max_num_models'])
    
    callbacks = []
    if cfg['swag']['enabled']:
        swag_callback = SWAGCallback(
            swag_model=swag_model,
            collect_every_n_steps=cfg['swag']['interval']
        )
        # Note: In a real scenario, you'd only start collecting after start_step
        # We can refine the callback to respect start_step
        callbacks.append(swag_callback)

    # 6. Pre-Distillation Evaluation
    print("\n[PRE-TRAIN] Running initial evaluation...")
    teacher_ppl = evaluate_ppl(teacher_model, eval_data, tokenizer, device, "Teacher")
    student_pre_ppl = evaluate_ppl(student_model, eval_data, tokenizer, device, "Student (Base)")

    # 7. Distillation Setup
    training_args = TrainingArguments(
        output_dir="./results_swag",
        per_device_train_batch_size=cfg['training']['batch_size'],
        gradient_accumulation_steps=cfg['training']['gradient_accumulation_steps'],
        num_train_epochs=cfg['training']['epochs'],
        learning_rate=float(cfg['training']['learning_rate']),
        fp16=cfg['training']['fp16'],
        logging_steps=50,
        save_strategy="no"
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

    # 8. Execution
    print("Starting SWAG Distillation...")
    trainer.train()

    # 9. Post-Distillation Evaluation
    print("\n[POST-TRAIN] Running final evaluation...")
    
    # Evaluate Standard Student (last weights)
    student_last_ppl = evaluate_ppl(student_model, eval_data, tokenizer, device, "Student (Last weights)")

    # Evaluate SWA Mean
    print("\nLoading SWA mean weights...")
    swag_model.get_mean_model()
    student_swa_ppl = evaluate_ppl(student_model, eval_data, tokenizer, device, "Student (SWA Mean)")

    # Evaluate a SWAG sample
    print("\nSampling from SWAG posterior...")
    swag_model.sample(scale=cfg['swag']['scale'])
    student_sample_ppl = evaluate_ppl(student_model, eval_data, tokenizer, device, "Student (SWAG Sample)")

    # Final Summary Table
    print(f"\n" + "="*40)
    print(f"   FINAL RESULTS SUMMARY (SWAG)")
    print(f"="*40)
    print(f"Teacher PPL:        {teacher_ppl:.2f}")
    print(f"Student (Before):   {student_pre_ppl:.2f}")
    print(f"Student (Last):     {student_last_ppl:.2f}")
    print(f"Student (SWA Mean): {student_swa_ppl:.2f}")
    print(f"Student (Sample):   {student_sample_ppl:.2f}")
    print(f"="*40)

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
