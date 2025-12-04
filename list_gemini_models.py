import os

import google.generativeai as genai

from logger import get_logger

logger = get_logger(__name__)

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    logger.warning("python-dotenv not found, trying to read .env manually")
    try:
        with open(".env", "r") as f:
            for line in f:
                if "=" in line:
                    key, value = line.strip().split("=", 1)
                    os.environ[key] = value
    except Exception as e:
        logger.exception(f"Error reading .env: {e}")

api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    logger.error("Error: GEMINI_API_KEY not found.")
else:
    logger.info(f"Found API Key: {api_key[:5]}...")
    try:
        genai.configure(api_key=api_key)
        logger.info("Listing available models...")
        for m in genai.list_models():
            if "generateContent" in m.supported_generation_methods:
                logger.info(f"Model: {m.name}")
    except Exception as e:
        logger.exception(f"Error listing models: {e}")
