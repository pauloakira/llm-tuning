# Python libs
import re
import json
from word2number import w2n
from fuzzywuzzy import fuzz

def standardizeResponse(response: str) -> str:
    # Convert response to lowercase
    response = response.lower()
    # Remove punctuation and special characters
    response = re.sub(r'[^a-z0-9\s]', '', response)
    # Attempt to convert written numbers to digits (optional, based on expected responses)
    try:
        response = " ".join([str(w2n.word_to_num(word)) if word.isalpha() else word for word in response.split()])
    except ValueError:
        # If conversion fails, pass the response as-is
        pass
    return response

def extract_relevant_answer(response: str, prompt: str) -> str:
    # Find the end of the prompt in the response and extract text after that
    prompt_end_index = response.find(prompt) + len(prompt)
    relevant_answer = response[prompt_end_index:].strip()
    return relevant_answer.split()[0]  # Assume the first word after the prompt is the answer


def levensteinSimilarity(response1: str, response2: str) -> int:
    return fuzz.ratio(response1, response2)

def evaluateResponse(llm_response: str, correct_answer: str, prompt: str, acceptance_threshold: int):
    # Extract the relevant part of the response
    relevant_llm_response = extract_relevant_answer(llm_response, prompt)
    
    # Standardize the extracted answer and the correct answer
    standardized_llm_response = standardizeResponse(relevant_llm_response)
    standardized_correct_answer = standardizeResponse(correct_answer)

    # Calculate Levenshtein distance (fuzzy match score)
    match_score = levensteinSimilarity(standardized_llm_response, standardized_correct_answer)

    # Check if the match score meets the acceptance threshold
    if match_score >= acceptance_threshold:
        return True, match_score  # Response is accepted as correct
    else:
        return False, match_score  # Response is not accepted as correct

