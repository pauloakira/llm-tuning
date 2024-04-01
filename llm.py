# Python libs
import time
import torch
import mlflow
from transformers import GPTNeoForCausalLM, GPT2Tokenizer

torch.backends.quantized.engine = 'qnnpack'  # For ARM CPUs

# Test Prompt
prompt = ("What is the capital of Brazil?")

def gpt_neo(qunatize: bool=False)-> str:
    model_path = "llm_gpt_neo"

    print("Loading GPT-Neo 1.3B model...")
    model = GPTNeoForCausalLM.from_pretrained("EleutherAI/gpt-neo-1.3B", cache_dir=model_path)
    tokenizer = GPT2Tokenizer.from_pretrained("EleutherAI/gpt-neo-1.3B", cache_dir=model_path)
    print("GPT-Neo 1.3B model loaded.")

    # Set the pad token id to the eos token id
    tokenizer.pad_token = tokenizer.eos_token

    # Quantize model
    if qunatize:
        model = torch.quantization.quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.float16)

    start_time = time.time()

    encoding = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True)
    input_ids = encoding["input_ids"]
    attention_mask = encoding["attention_mask"]

    gen_tokens = model.generate(
        input_ids,
        attention_mask=attention_mask,
        pad_token_id=tokenizer.eos_token_id,
        do_sample=True,  # Enable sampling to introduce randomness
        temperature=0.1,  # Adjust temperature to balance randomness and determinism
        max_length=30,  # Adjust max_length to fit your needs
        top_k=50,  # Consider adjusting top_k
        top_p=0.95,  # Consider adjusting top_p
        num_return_sequences=1,  # Set the number of responses you want
    )

    gen_text = tokenizer.batch_decode(gen_tokens)[0]

    execution_time = time.time() - start_time

    return gen_text, execution_time

if __name__ == "__main__":
    # Parameters setup
    quantize = False
    do_sample = True
    temperature = 0.1
    max_length = 30
    top_k = 50
    top_p = 0.95
    num_return_sequences = 1

    response, execution_time = gpt_neo()

    with mlflow.start_run():
        mlflow.log_param("quantize", quantize)
        mlflow.log_param("do_sample", do_sample)
        mlflow.log_param("temperature", temperature)
        mlflow.log_param("max_length", max_length)
        mlflow.log_param("top_k", top_k)
        mlflow.log_param("top_p", top_p)
        mlflow.log_param("num_return_sequences", num_return_sequences)
        mlflow.log_metric("execution_time", execution_time)
        mlflow.log_text("response", response)

        
