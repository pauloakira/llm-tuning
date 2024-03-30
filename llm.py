# Python libs
from transformers import GPTNeoForCausalLM, GPT2Tokenizer

def gpt_neo():
    model_path = "llm_gpt_neo"

    print("Loading GPT-Neo 1.3B model...")
    model = GPTNeoForCausalLM.from_pretrained("EleutherAI/gpt-neo-1.3B", cache_dir=model_path)
    tokenizer = GPT2Tokenizer.from_pretrained("EleutherAI/gpt-neo-1.3B", cache_dir=model_path)
    print("GPT-Neo 1.3B model loaded.")

    prompt = (
        "Who are you?"
    )

    input_ids = tokenizer(prompt, return_tensors="pt").input_ids

    gen_tokens = model.generate(
        input_ids,
        do_sample=True,
        temperature=0.1,
        max_length=100,
    )

    gen_text = tokenizer.batch_decode(gen_tokens)[0]

    return gen_text

if __name__ == "__main__":
    print(gpt_neo())