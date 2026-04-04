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
from src.trainer import SGLDDistillationTrainer
from src.utils import load_config, WeightCollectionCallback, evaluate_ppl

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

    # Tokenize all splits (train, test, validation)
    tokenized_datasets = dataset.map(tokenize_fn, batched=True, remove_columns=["text"])
    
    # Filter out empty lines and create splits
    train_data = tokenized_datasets["train"].filter(lambda x: len(x["input_ids"]) > 0)
    
    # Define eval_data (using the 'test' split from wikitext)
    # We take a subset (e.g., 100 samples) for faster evaluation during training
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

    # 5. Prepare Weight Collection Callback
    callbacks = []
    if cfg['sampling']['enabled']:
        steps_to_collect = list(range(
            cfg['sampling']['start_step'], 
            cfg['sampling']['end_step'] + 1, 
            cfg['sampling']['interval']
        ))
        weight_callback = WeightCollectionCallback(
            collection_steps=steps_to_collect,
            save_dir=cfg['sampling']['save_dir']
        )
        callbacks.append(weight_callback)

    # 6. Pre-Distillation Evaluation
    print("\n[PRE-TRAIN] Running initial evaluation...")
    teacher_ppl = evaluate_ppl(teacher_model, eval_data, tokenizer, device, "Teacher")
    student_pre_ppl = evaluate_ppl(student_model, eval_data, tokenizer, device, "Student (Base)")
    # 7. Distillation Setup
    training_args = TrainingArguments(
        output_dir="./results",
        per_device_train_batch_size=cfg['training']['batch_size'],
        gradient_accumulation_steps=cfg['training']['gradient_accumulation_steps'],
        num_train_epochs=cfg['training']['epochs'],
        learning_rate=float(cfg['training']['learning_rate']),
        fp16=cfg['training']['fp16'],
        logging_steps=50,
        evaluation_strategy="steps", # Now it will use eval_data
        eval_steps=500,               # Evaluate every 500 steps
        save_strategy="no"
    )

    trainer = SGLDDistillationTrainer(
        model=student_model,
        teacher_model=teacher_model,
        temperature=cfg['training']['temperature'],
        alpha=cfg['training']['alpha'],
        args=training_args,
        train_dataset=train_data,
        eval_dataset=eval_data,      # <--- Now properly defined
        callbacks=callbacks,
        data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
    )

    # 8. Execution
    print("Starting SGLD Distillation...")
    trainer.train()
    # 9. Post-Distillation Evaluation
    print("\n[POST-TRAIN] Running final evaluation...")
    student_post_ppl = evaluate_ppl(student_model, eval_data, tokenizer, device, "Distilled Student")

    # Final Summary Table
    print(f"\n" + "="*30)
    print(f"   FINAL RESULTS SUMMARY")
    print(f"="*30)
    print(f"Teacher PPL:      {teacher_ppl:.2f}")
    print(f"Student (Before): {student_pre_ppl:.2f}")
    print(f"Student (After):  {student_post_ppl:.2f}")
    print(f"="*30)
    # 10. Save results
    student_model.save_pretrained(cfg['save_path'])
    tokenizer.save_pretrained(cfg['save_path'])
    print(f"Model saved to {cfg['save_path']}")

if __name__ == "__main__":
    main()