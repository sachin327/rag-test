from services.upload_service import UploadService
from services.generate_question_service import GenerateQuestionService

UPLOAD_SERVICE = None
GENERATE_QUESTION_SERVICE = None


def get_upload_service():
    global UPLOAD_SERVICE
    if not UPLOAD_SERVICE:
        UPLOAD_SERVICE = UploadService()
    return UPLOAD_SERVICE


def get_generate_question_service():
    global GENERATE_QUESTION_SERVICE
    if not GENERATE_QUESTION_SERVICE:
        GENERATE_QUESTION_SERVICE = GenerateQuestionService()
    return GENERATE_QUESTION_SERVICE
