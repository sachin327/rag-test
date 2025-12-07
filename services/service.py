from services.upload_service import UploadService

UPLOAD_SERVICE = None


def get_upload_service():
    global UPLOAD_SERVICE
    if not UPLOAD_SERVICE:
        UPLOAD_SERVICE = UploadService()
    return UPLOAD_SERVICE
