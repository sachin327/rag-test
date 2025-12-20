import time
from functools import wraps
import json
from logger import get_logger
from typing import Dict, Any

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


def extract_payload(obj: Dict[str, Any]) -> Dict[str, Any]:
    # If it looks like a schema wrapper with a "data" field, unwrap it
    schemaish = {
        "$defs",
        "properties",
        "required",
        "title",
        "type",
        "additionalProperties",
    }
    if "data" in obj and schemaish.intersection(obj.keys()):
        return obj["data"]
    return obj


def safe_str_to_json(string):
    try:
        result = json.loads(string)
        return extract_payload(result)
    except Exception as e:
        logger.warning(f"Failed to convert string to JSON: {e}")
        print("##########")
        print(string)
        print("##########")
        return string
