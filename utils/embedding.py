from typing import List
import requests
from dotenv import load_dotenv
from logger import get_logger

load_dotenv()
logger = get_logger(__name__)


class Embedding:
    def __init__(self, api_url: str = "http://0.0.0.0:9996/embedding"):
        self.api_url = api_url
        logger.info(f"Initialized Embedding with API URL: {self.api_url}")

    def embed(self, texts: List[str]) -> List[List[float]]:
        """Embed a list of texts using the external API."""
        try:
            response = requests.post(self.api_url, json=texts)
            response.raise_for_status()
            embeddings = response.json()
            return embeddings
        except requests.exceptions.RequestException as e:
            logger.error(f"Error calling embedding API: {e}")
            raise e
        except Exception as e:
            logger.error(f"Unexpected error during embedding: {e}")
            raise e
