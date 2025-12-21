from services.upload_service import UploadService
from services.generate_question_service import GenerateQuestionService
from services.query_service import QueryService
from services.llm_insights import LLMInsightsService

UPLOAD_SERVICE = None
GENERATE_QUESTION_SERVICE = None
QUERY_SERVICE = None
LLM_INSIGHTS_SERVICE = None


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


def get_query_service():
    global QUERY_SERVICE
    if not QUERY_SERVICE:
        QUERY_SERVICE = QueryService()
    return QUERY_SERVICE


def get_llm_insights_service():
    global LLM_INSIGHTS_SERVICE
    if not LLM_INSIGHTS_SERVICE:
        LLM_INSIGHTS_SERVICE = LLMInsightsService()
    return LLM_INSIGHTS_SERVICE
