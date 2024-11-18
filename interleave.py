from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import torch
# Use a pipeline as a high-level helper
from transformers import pipeline

messages = [
    {"role": "user", "content": "Who are you?"},
]
pipe = pipeline("text-generation", model="lmms-lab/llava-next-interleave-qwen-7b-dpo")
pipe(messages)


model_name = "lmms-lab/llava-next-interleave-qwen-7b"

def load_model_and_tokenizer():
    # Load the model      
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True
    )
    
    # Load the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True
    )
    
    return model, tokenizer

def generate_response_with_model(model, tokenizer, prompt):
    messages = [
        {"role": "user", "content": prompt}
    ]
    
    # Apply chat template
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    
    # Tokenize input
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    
    # Generate response
    with torch.no_grad():
        output = model.generate(**inputs, max_new_tokens=512)
    
    # Decode and return response
    response = tokenizer.decode(output[0], skip_special_tokens=True)
    return response

def generate_response_with_pipeline(prompt):
    pipe = pipeline("text-generation", model=model_name, trust_remote_code=True)
    messages = [
        {"role": "user", "content": prompt}
    ]
    return pipe(messages)

if __name__ == "__main__":
    print("Loading model and tokenizer...")
    model, tokenizer = load_model_and_tokenizer()
    print("Model and tokenizer loaded successfully!")

    # Example usage with direct model interaction
    prompt = "Translate the following English text to French: 'Hello, how are you?'"
    print("\nUsing direct model interaction:")
    response = generate_response_with_model(model, tokenizer, prompt)
    print("Response:", response)

    # Example usage with pipeline
    print("\nUsing pipeline:")
    pipeline_response = generate_response_with_pipeline(prompt)
    print("Pipeline Response:", pipeline_response)