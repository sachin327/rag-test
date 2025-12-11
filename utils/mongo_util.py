from bson.objectid import ObjectId
from typing import List

from db.mongo_db import MongoDB
from logger import get_logger

logger = get_logger(__name__)


def get_topics_mongo(mongo: MongoDB, collection_name: str, topic_ids: List[str]):
    try:
        # 1. Get existing topics for this subject_id
        topics = mongo.get_collection(collection_name).find(
            {"_id": {"$in": [ObjectId(topic_id) for topic_id in topic_ids]}}
        )
        topics = [
            {"title": topic.get("title"), "description": topic.get("description")}
            for topic in topics
        ]
        return topics
    except Exception as e:
        logger.error(f"Error getting docs from mongo: {e}")
        return []


def get_chapters_mongo(mongo: MongoDB, collection_name: str, chapter_ids: List[str]):
    try:
        # 1. Get existing topics for this subject_id
        chapters = mongo.get_collection(collection_name).find(
            {"_id": {"$in": [ObjectId(chapter_id) for chapter_id in chapter_ids]}}
        )
        chapters = [{"name": chapter.get("title")} for chapter in chapters]
        return chapters
    except Exception as e:
        logger.error(f"Error getting docs from mongo: {e}")
        return []


def get_subjects_mongo(mongo: MongoDB, collection_name: str, subject_ids: List[str]):
    try:
        # 1. Get existing topics for this subject_id
        subjects = mongo.get_collection(collection_name).find(
            {"_id": {"$in": [ObjectId(subject_id) for subject_id in subject_ids]}}
        )
        subjects = [{"name": subject.get("title")} for subject in subjects]
        return subjects
    except Exception as e:
        logger.error(f"Error getting docs from mongo: {e}")
        return []


def get_classes_mongo(mongo: MongoDB, collection_name: str, class_ids: List[str]):
    try:
        # 1. Get existing topics for this subject_id
        classes = mongo.get_collection(collection_name).find(
            {"_id": {"$in": [ObjectId(class_id) for class_id in class_ids]}}
        )
        classes = [{"name": classe.get("title")} for classe in classes]
        return classes
    except Exception as e:
        logger.error(f"Error getting docs from mongo: {e}")
        return []
