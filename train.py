import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from datasets import load_dataset
from peft import get_peft_model, LoraConfig
from trl import SFTTrainer

def load_model_and_tokenizer(model_id: str):
    """Load the base model and tokenizer with 8-bit quantization support."""
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        load_in_8bit=True,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True
    )

    return model, tokenizer


def apply_lora(model):
    """Attach a LoRA adapter to the model for parameter-efficient fine-tuning."""
    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "fc1", "fc2"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )

    return get_peft_model(model, peft_config)


def load_and_prepare_dataset(path: str):
    """Load and format the dataset from a JSONL file."""
    dataset = load_dataset("json", data_files=path)

    def format_example(example):
        return {
            "text": f"<|user|>\n{example['instruction']}\n<|assistant|>\n{example['response']}"
        }

    return dataset["train"].map(format_example)


def train(model, tokenizer, train_data):
    """Train the model using TRL's SFTTrainer."""
    training_args = TrainingArguments(
        output_dir="./sarcastic-phi3-output",
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        num_train_epochs=3,
        logging_steps=10,
        save_strategy="epoch",
        learning_rate=2e-4,
        bf16=True,
        optim="paged_adamw_8bit",
        warmup_steps=20,
        save_total_limit=2,
        report_to="none"
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_data,
        args=training_args
    )

    trainer.train()
    return trainer


def save_model(trainer, output_path: str):
    """Save the trained model and tokenizer."""
    trainer.save_model(output_path)
    trainer.tokenizer.save_pretrained(output_path)
    trainer.model.save_pretrained(output_path)


if __name__ == "__main__":
    model_id = "microsoft/phi-3-mini-128k-instruct"
    data_path = "./data/sarcasm_data_gemini.jsonl"
    output_dir = "./sarcastic-phi3-model"

    model, tokenizer = load_model_and_tokenizer(model_id)
    model = apply_lora(model)
    train_data = load_and_prepare_dataset(data_path)
    trainer = train(model, tokenizer, train_data)
    save_model(trainer, output_dir)

