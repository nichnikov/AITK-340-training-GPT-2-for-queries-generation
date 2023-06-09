from transformers import AutoModelForCausalLM, TrainingArguments
from peft import get_peft_config, get_peft_model, LoraConfig, TaskType
model_name_or_path = "bigscience/mt0-large"
tokenizer_name_or_path = "bigscience/mt0-large"


peft_config = LoraConfig(
    task_type=TaskType.SEQ_2_SEQ_LM, inference_mode=False, r=8, lora_alpha=32, lora_dropout=0.1
)

model = AutoModelForCausalLM.from_pretrained("ai-forever/rugpt3small_based_on_gpt2")
model = get_peft_model(model, peft_config)
model.print_trainable_parameters()
model.active_peft_config

training_args = TrainingArguments(
    "finetune_gpt3",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    num_train_epochs=1,
    lr_scheduler_type='cosine',
    logging_dir='./logs',
    load_best_model_at_end=True,
    per_device_train_batch_size=1,  # adjust this according to your GPU memory
    per_device_eval_batch_size=1,  # adjust this according to your GPU memory
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
)