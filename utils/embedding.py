from typing import List
from dotenv import load_dotenv
from logger import get_logger

load_dotenv()
logger = get_logger(__name__)

class Embedding:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model_name = model_name
        self._model = None
        self._model_type = None
        self.init_model()

    def init_model(self):
        """Load the embedding model."""
            

        logger.info(f"Loading Embd model '{self.model_name}'...")

        # Map common short names to FastEmbed supported names
        fastembed_model_name = self.model_name
        if self.model_name == "all-MiniLM-L6-v2":
            fastembed_model_name = "sentence-transformers/all-MiniLM-L6-v2"

        try:
            from fastembed import TextEmbedding

            self._model = TextEmbedding(model_name=fastembed_model_name)
            self._model_type = "fastembed"
            logger.info(
                f"FastEmbed model '{fastembed_model_name}' loaded successfully."
            )
        except ValueError:
            logger.warning(
                f"Model '{fastembed_model_name}' not found in FastEmbed. Switching to default fast model..."
            )

            self._model = TextEmbedding()
            self._model_type = "fastembed"
            logger.info(
                "Loaded default FastEmbed model (BAAI/bge-small-en-v1.5)."
            )
        except ImportError:
            logger.info(
                "FastEmbed not installed. For faster loading, install it: pip install fastembed"
            )
            logger.info("Falling back to SentenceTransformers...")
            self._load_sentence_transformer()
        except Exception as e:
            logger.error(
                f"Error loading embedding model: {e}"
            )
            raise e

        return self._model

    def _load_sentence_transformer(self):
        """Helper to load SentenceTransformer."""
        try:
            import torch
            from sentence_transformers import SentenceTransformer

            device = "cuda" if torch.cuda.is_available() else "cpu"
            logger.info(
                f"Loading SentenceTransformer model '{self.model_name}' on {device}..."
            )
            self._model = SentenceTransformer(self.model_name, device=device)
            self._model_type = "sentence_transformers"
            logger.info("Model loaded successfully.")
        except ImportError:
            raise ImportError(
                "Neither 'fastembed' nor 'sentence-transformers' is installed. Please install one."
            )

    def embed(self, texts: List[str]) -> List[List[float]]:
        """Embed a list of texts."""
        if self._model_type == "fastembed":
            return list(self._model.embed(texts))
        else:
            return self._model.encode(texts).tolist()

embedding = Embedding()