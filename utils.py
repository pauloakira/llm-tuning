# Python libs
import time

def measureExecTime(func):
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        execution_time = end - start
        print(f"Function {func.__name__} took {execution_time} seconds to run.")
        return result
    return wrapper