import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig
)
from peft import PeftModel


def load_model(base_model_id: str, adapter_path: str, offload_folder: str = "./offload"):
    """Loads the base model and LoRA adapter with 8-bit quantization."""
    print("Loading base model...")
    bnb_config = BitsAndBytesConfig(load_in_8bit=True)

    model = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        device_map="auto",
        torch_dtype=torch.float16,
        offload_folder=offload_folder,
        quantization_config=bnb_config
    )

    print("Applying LoRA adapter...")
    model = PeftModel.from_pretrained(model, adapter_path)

    return model


def load_tokenizer(model_id: str):
    """Loads the tokenizer for the model."""
    return AutoTokenizer.from_pretrained(model_id)


def generate_response(model, tokenizer, prompt: str, device: str = "cuda", max_new_tokens: int = 500) -> str:
    """Generates a response from the model given a prompt."""
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    output = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=1.0,
        top_p=1.0
    )

    decoded = tokenizer.decode(output[0], skip_special_tokens=True)
    return decoded


def chat_loop(model, tokenizer, device: str):
    """Runs an interactive chat loop."""
    print("Chatbot loaded! Type your message or 'exit' to quit.\n")

    while True:
        user_input = input("You: ")
        if user_input.lower() in ["exit", "quit"]:
            print("Goodbye!")
            break

        prompt = f"<|user|>\n{user_input}\n<|assistant|>\n"
        response = generate_response(model, tokenizer, prompt, device)
        reply = response.split("<|assistant|>")[-1].strip()
        print("Bot:", reply)


def main():
    base_model_id = "microsoft/phi-3-mini-128k-instruct"
    adapter_path = "./sarcastic-phi3-model"

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = load_model(base_model_id, adapter_path)
    tokenizer = load_tokenizer(base_model_id)

    model.to(device)
    chat_loop(model, tokenizer, device)


if __name__ == "__main__":
    main()

