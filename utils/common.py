import time
from functools import wraps

from logger import get_logger

logger = get_logger(__name__)


def timing_decorator(func):
    """Decorator to measure function execution time."""

    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        logger.debug(f"[TIME] {func.__name__} took {end_time - start_time:.4f} seconds")
        return result

    return wrapper
