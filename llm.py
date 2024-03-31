# Python libs
import torch
import mlflow
from transformers import GPTNeoForCausalLM, GPT2Tokenizer

# Custom libs
from utils import measureExecTime

torch.backends.quantized.engine = 'qnnpack'  # For ARM CPUs

# Test Prompt
prompt = ("What is the capital of Brazil?")

@measureExecTime
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

    return gen_text

if __name__ == "__main__":
    
    # Setup MLFlow
    run_id = "gpt_neo"
    run = mlflow.start_run(run_id=run_id)
    print(gpt_neo())

    mlflow.end_run()