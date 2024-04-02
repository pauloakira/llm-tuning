# Python libs
from typing import List
from dataclasses import dataclass

@dataclass
class Evaluation:
    prompt: str
    expected_completion: str
    response: str
    match_score: int
    is_correct: bool
    execution_time: float