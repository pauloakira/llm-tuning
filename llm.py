# Python libs
import time
import json
import torch
import mlflow
import numpy as np
from typing import List
from transformers import GPTNeoForCausalLM, GPT2Tokenizer

# Custom libs
from models import Evaluation
from performance import evaluateResponse

torch.backends.quantized.engine = 'qnnpack'  # For ARM CPUs

def gpt_neo(prompt:str, quantize: bool=False)-> str:
    model_path = "llm_gpt_neo"

    print("Loading GPT-Neo 1.3B model...")
    model = GPTNeoForCausalLM.from_pretrained("EleutherAI/gpt-neo-1.3B", cache_dir=model_path)
    tokenizer = GPT2Tokenizer.from_pretrained("EleutherAI/gpt-neo-1.3B", cache_dir=model_path)
    print("GPT-Neo 1.3B model loaded.")

    # Set the pad token id to the eos token id
    tokenizer.pad_token = tokenizer.eos_token

    # Quantize model
    if quantize:
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

def evaluate()-> List[Evaluation]:
    with open("assets/questions.json", "r") as file:
        data = json.load(file)

    evaluation_list = []
    
    for item in data:
            start_time = time.time()
            prompt = item["prompt"]
            response, execution_time = gpt_neo(prompt)
            correct_answer = item['expected_completion']
            is_correct, match_score = evaluateResponse(response, correct_answer, prompt, 70)
            print(f"Prompt: {prompt}")
            print(f"Response: {response}")
            print(f"Correct answer: {correct_answer}")
            print(f"Match score: {match_score}")
            print(f"Is correct: {is_correct}")
            print(f"Execution time: {execution_time}")
            print("-------------------\n")

            elapsed_time = time.time() - start_time

            evaluation = Evaluation(prompt=prompt, 
                                    expected_completion=correct_answer, 
                                    response=response, 
                                    match_score=match_score, 
                                    is_correct=bool(is_correct), 
                                    execution_time=elapsed_time)

            evaluation_list.append(evaluation)

    return evaluation_list

if __name__ == "__main__":
    # Set MLFlow experiment
    mlflow.set_experiment("LLM Evaluation")

    # Evaluate the model
    evaluations = evaluate()

    # Generate timestamp
    timestamp = int(time.time()*1000)

    with mlflow.start_run():
         # Define a name for the run
        mlflow.set_tag("mlflow.runName", f"LLM-Evaluation-{timestamp}")
        scores = np.array([item.match_score for item in evaluations])
        total_score = np.sum(scores) / len(scores)
        std_dev = np.std(scores)
        mlflow.log_metric({"total_score": total_score, "std_dev": std_dev})
        # Iterate through each evaluation and log its details
        for index, item in enumerate(evaluations, start=1):
            # Log parameters for each item as a dictionary
            # To avoid overwriting, prefix or suffix each key with a unique identifier (e.g., the index)
            mlflow.log_params({
                f"prompt_{index}": item.prompt,
                f"expected_completion_{index}": item.expected_completion,
                f"response_{index}": item.response
            })
            
            # Convert boolean to float for the "is_correct" metric
            is_correct_metric = 1.0 if item.is_correct else 0.0
            # Log metrics for each item, also with a unique identifier
            mlflow.log_metrics({
                f"match_score_{index}": item.match_score,
                f"execution_time_{index}": item.execution_time,
                f"is_correct_{index}": is_correct_metric
            })

    print("Evaluation completed.")
    

        
