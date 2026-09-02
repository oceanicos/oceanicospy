import time
import logging
from functools import wraps

def timing_decorator(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.perf_counter()  # high-resolution timer
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        duration = end_time - start_time
        message = f"Function '{func.__name__}' took {duration:.4f} seconds to run."
        if args and hasattr(args[0], "__dict__"):
            self = args[0]
            if hasattr(self, "logger"):
                self.logger.info(message)
            else:
                print(message)
        else:
            print(message)
        return result
    return wrapper