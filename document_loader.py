import docx
import pypdf

from logger import get_logger

# Initialize logger
logger = get_logger(__name__)


class DocumentLoader:
    """Mocks loading content from various document types into raw text."""

    @staticmethod
    def _load_txt(file_path: str) -> str:
        """Loads content from a standard text file."""
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()
        return ""

    @staticmethod
    def _load_pdf(file_path: str) -> str:
        """Loads content from a PDF document using pypdf."""
        reader = pypdf.PdfReader(file_path)
        text = ""
        for page in reader.pages:
            text += page.extract_text() + "\n"
        return text

    @staticmethod
    def _load_docx(file_path: str) -> str:
        """Loads content from a Word document using python-docx."""
        document = docx.Document(file_path)
        return "\n".join([paragraph.text for paragraph in document.paragraphs])

    @staticmethod
    def load_document(file_path: str) -> str:
        """Simulates loading content based on file extension (PDF, TXT,
        DOCX)."""
        logger.info(f"Loading document: {file_path}")
        try:
            if file_path.endswith(".pdf"):
                return DocumentLoader._load_pdf(file_path)
            elif file_path.endswith((".txt", ".text")):
                return DocumentLoader._load_txt(file_path)
            elif file_path.endswith((".doc", ".docx")):
                return DocumentLoader._load_docx(file_path)
            else:
                logger.error(f"Invalid document type: {file_path}")
                raise ValueError("Invalid document type")
        except Exception as e:
            logger.exception(f"Error loading document {file_path}: {e}")
            raise


if __name__ == "__main__":
    # Example usage
    try:
        result = DocumentLoader.load_document("data/sample.pdf")
        logger.info(f"Extracted text: {result}")
        pass
    except Exception as e:
        logger.exception(e)
