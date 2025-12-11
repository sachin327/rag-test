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
            {
                "title": topic.get("title"),
                "description": topic.get("description"),
                "id": topic.get("_id").__str__(),
            }
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


def get_topics_from_subject_mongo(
    mongo: MongoDB, collection_name: str, subject_id: str = ""
):
    try:
        # 1. Get existing topics for this subject_id
        pipeline = [
            # 1. Match/Filter: Find documents in chapterTopics by subjectId
            {"$match": {"subjectId": ObjectId(subject_id)}},
            # 2. Lookup/Join: Join chapterTopics with the topics collection
            {
                "$lookup": {
                    "from": "topics",  # The collection to join (the 'topics' collection)
                    "localField": "topicId",  # Field from the input documents (chapterTopics)
                    "foreignField": "_id",  # Field from the documents of the 'from' collection (topics)
                    "as": "topicDetails",  # The name for the new array field in the output documents
                }
            },
            # 3. Unwind (Optional but Recommended): Deconstruct the 'topicDetails' array.
            # Since topicId is likely unique, this turns the 'topicDetails' array
            # (which contains one document) into a single object.
            {"$unwind": "$topicDetails"},
            # 4. Project: Shape the final output to include only the fields you need
            {
                "$project": {
                    "_id": 0,  # Exclude the _id field
                    "topicId": "$topicId",  # Keep the topicId
                    # Get the title from the joined document
                    "title": "$topicDetails.title",
                }
            },
        ]

        existing_docs = list(mongo.get_collection(collection_name).aggregate(pipeline))
        return existing_docs
    except Exception as e:
        logger.error(f"Error getting docs from mongo: {e}")
        return None
