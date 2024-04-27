# Python libs
import time
import json
import torch
import mlflow
import numpy as np
from typing import List
from mlx_lm import load, generate
from transformers import GPTNeoForCausalLM, GPT2Tokenizer, AutoTokenizer, FalconForCausalLM, AutoModelForCausalLM

# Custom libs
from models import Evaluation
from performance import evaluateResponseWithContinuousScore, checkResponse

torch.backends.quantized.engine = 'qnnpack'  # For ARM CPUs

def load_model(model_name: str)-> torch.nn.Module:
    if model_name == "gpt_neo":
        model_path = "llm_gpt_neo"
        print("Loading GPT-Neo model...")
        model = GPTNeoForCausalLM.from_pretrained("EleutherAI/gpt-neo-1.3B", cache_dir=model_path)
        tokenizer = GPT2Tokenizer.from_pretrained("EleutherAI/gpt-neo-1.3B", cache_dir=model_path)
        print("Model loaded.")
    elif model_name == "falcon_1b":
        model_path = "llm_falcon"
        print("Loading Falcon 1B model...")
        model = FalconForCausalLM.from_pretrained("Rocketknight1/falcon-rw-1b", cache_dir=model_path)
        tokenizer = AutoTokenizer.from_pretrained("Rocketknight1/falcon-rw-1b", cache_dir=model_path)
        print("Model loaded.")
    elif model_name == "falcon_7b":
        model_path = "llm_falcon7b"
        print("Loading Falcon 7B model...")
        model = AutoModelForCausalLM.from_pretrained("tiiuae/falcon-7b", cache_dir=model_path)
        tokenizer = AutoTokenizer.from_pretrained("tiiuae/falcon-7b", cache_dir=model_path)
        print("Model loaded.")
    elif model_name == "mistral_7b":
        model_path = "llm_mistral7b"
        print("Loading Mistral 7B model...")
        model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2", cache_dir=model_path)
        tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2", cache_dir=model_path)
    else:
        raise ValueError("Invalid model name.")
    
    return model, tokenizer

def load_model_with_mlx(model_name: str)-> torch.nn.Module:
    if model_name == "mistral_7b":
        print("Loading Mistral 7B model with MLX...")
        model, tokenizer = load("mistralai/Mistral-7B-Instruct-v0.2")
        print("Model loaded.")
    if model_name == "gemma-2b-it":
        print("Loading Gemma 2B IT model with MLX...")
        model, tokenizer = load("mlx-community/quantized-gemma-2b-it")
        print("Model loaded.")
    if model_name == "qwen-0.5b-chat":
        print("Loading Qwen 1.5 0.5B Chat model with MLX...")
        model, tokenizer = load("mlx-community/Qwen1.5-0.5B-Chat")
        print("Model loaded.")
    if model_name == "llama3-8b":
        print("Loading Llama3 8B model with MLX...")
        model, tokenizer = load("mlx-community/Meta-Llama-3-8B-Instruct")
        print("Model loaded.")
    else:
        raise ValueError("Invalid model name.")
    
    return model, tokenizer

def generate_response(model: torch.nn.Module, tokenizer: torch.nn.Module, prompt: str, quantize: bool = False)-> str:
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
        max_length=550,  # Adjust max_length to fit your needs
        top_k=10,  # Consider adjusting top_k
        top_p=0.95,  # Consider adjusting top_p
        num_return_sequences=1,  # Set the number of responses you want
    )

    gen_text = tokenizer.batch_decode(gen_tokens)[0]

    execution_time = time.time() - start_time

    return gen_text, execution_time

def evaluate(model_name: str = "gpt_neo")-> List[Evaluation]:
    with open("assets/questions.json", "r") as file:
        data = json.load(file)

    evaluation_list = []

    model, tokenizer = load_model(model_name)
    
    for item in data:
            start_time = time.time()
            prompt = item["prompt"]
            response, execution_time = generate_response(model, tokenizer, prompt)
            correct_answer = item['expected_completion']
            is_correct, match_score = evaluateResponseWithContinuousScore(response, correct_answer, prompt, 70)
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

def mlflowPipeline(model_name: str = "gpt_neo"):
    # Set MLFlow experiment
    mlflow.set_experiment("LLM Evaluation")

    # Evaluate the model
    evaluations = evaluate(model_name=model_name)

    # Generate timestamp
    timestamp = int(time.time()*1000)

    with mlflow.start_run():
         # Define a name for the run
        mlflow.set_tag("mlflow.runName", f"LLM-Evaluation-{model_name}-{timestamp}")
        scores = np.array([item.match_score for item in evaluations])
        total_score = np.sum(scores) / len(scores)
        std_dev = np.std(scores)
        mlflow.log_metrics({"total_score": total_score, "std_dev": std_dev})
        # Iterate through each evaluation and log its details
        for index, item in enumerate(evaluations, start=1):
            # Log parameters for each item as a dictionary
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

def evaluateMLXModels(model_name: str = "mistral_7b")-> float:
    with open("assets/questions.json", "r") as file:
        data = json.load(file)

    model, tokenizer = load_model_with_mlx(model_name)
    correct_count = 0
    for item in data:
        prompt = item["prompt"]
        response = generate(model, tokenizer, prompt=prompt, verbose=False)
        print(f"Prompt: {prompt}")
        print(f"Response: {response}")
        correct_answer = item['expected_completion']
        is_correct = checkResponse(response, correct_answer)
        print(f"-----Is the answer correct?: {is_correct}")
    
        if is_correct:
            correct_count += 1
    
    total_score = correct_count / len(data)

    return total_score


if __name__ == "__main__":
    
    prompt = """
    <s>[INST]Can you classify this abstract: "The quality of texts generated by natural language generation (NLG) systems is hard to measure automatically. Conventional reference-based metrics, such as BLEU and ROUGE, have been shown to have relatively low correlation with human judgments, especially for tasks that require creativity and diversity. Recent studies suggest using large language models (LLMs) as reference-free metrics for NLG evaluation, which have the benefit of being applicable to new tasks that lack human references. However, these LLM-based evaluators still have lower human correspondence than medium-size neural evaluators. In this work, we present G-Eval, a framework of using large language models with chain-of-thoughts (CoT) and a form-filling paradigm, to assess the quality of NLG outputs. We experiment with two generation tasks, text summarization and dialogue generation. We show that G-Eval with GPT-4 as the backbone model achieves a Spearman correlation of 0.514 with human on summarization task, outperforming all previous methods by a large margin. We also propose preliminary analysis on the behavior of LLM-based evaluators, and highlight the potential issue of LLM-based evaluators having a bias towards the LLM-generated texts." in one of the following categories: "Machine Learning", "Deep Learning", "Natural Language Processing", "Computer Vision", "High Dimensional PDE", "Nonlinear PDE", "Transformers", "Optimization", "Artificial Neural Network", "Multi-modal Large Language Model", "Linear Operators", "Nonlinear Operators", "Synthetic Data", "Large Language Model", "Prompt Engineering", "Inference Techniques", "Convolutional Neural Network", "Probabilistic Models", "Physics-informed Neural Network", "Memory Mechanisms in LLMs", "Retrieval Augmented Generation", "Fine-tuning Strategies", "Quantization", "Small Language Models", "Linear Optimization", "Nonlinear Optimization", "Ethics and Fairness in AI", "Healthcare Applications of AI", "Robotics and Autonomous Systems", "Finance and Economics Applications", "Legal and Ethical AI", "Quantum Machine Learning", "Federated Learning", "AI for Climate Change", "Open-Source AI Projects", "Platform-Specific Development", "Evaluation and Benchmarks"?
    
    Intructions:
    -Give only the final classification
    -Format the output in a Python list. For example: ["Natural Language Processing"]
    -Choose only one category[/INST]"""

    # model, tokenizer = load_model("mistral_7b")
    # response, exec_time = generate_response(model, tokenizer, prompt)
    # print(response)

    # model, tokenizer = load_model_with_mlx("mistral_7b")
    # model, tokenizer = load_model_with_mlx("gemma-2b-it")
    model, tokenizer = load_model_with_mlx("llama3-8b")
    # response = generate(model, tokenizer, prompt="Who are you?", verbose=False)
    # print(response)
    response = generate(model, tokenizer, prompt=prompt, verbose=False)
    print(response)

    # score = evaluateMLXModels("gemma-2b-it")
    # print(f"Score for Gemma 2B IT: {score}")
    
