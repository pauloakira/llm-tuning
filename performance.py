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

def levensteinSimilarity(response1: str, response2: str) -> int:
    return fuzz.ratio(response1, response2)

def evaluateResponse(llm_response: str, correct_answer: str, acceptance_threshold):
    # Standardize responses
    standardized_llm_response = standardizeResponse(llm_response)
    standardized_correct_answer = standardizeResponse(correct_answer)
     # Calculate Levenshtein distance (fuzzy match score)
    match_score = levensteinSimilarity(standardized_llm_response, standardized_correct_answer)

    # Check if the match score meets the acceptance threshold
    if match_score >= acceptance_threshold:
        return True, match_score  # Response is accepted as correct
    else:
        return False, match_score  # Response is not accepted as correct
