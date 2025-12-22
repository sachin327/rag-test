import os
from bson import ObjectId
from dotenv import load_dotenv

from db.mongo_db import MongoDB
from llm.llm_open_router import LLMService
from logger import get_logger

logger = get_logger(__name__)
load_dotenv()


class LLMInsightsService:
    def __init__(self):
        try:
            self.mongo = MongoDB(os.getenv("MONGO_URI"), os.getenv("MONGO_DB_NAME"))
            self.llm = LLMService()
        except Exception as e:
            logger.exception(f"Error initializing LLMInsightsService services: {e}")
            raise

    def get_ai_insights_subject(self, subject_id: str, user_id: str) -> str:
        try:
            try:
                subject_id_oid = ObjectId(subject_id)
                user_id_oid = ObjectId(user_id)
            except Exception as e:
                return f"Invalid ID format: {e}"

            logger.info(
                f"Fetching progress for Subject ID: {subject_id_oid} and User ID: {user_id_oid}..."
            )

            # Fetch Subject Name
            subjects_coll = self.mongo.get_collection("subjects")
            subject_doc = subjects_coll.find_one({"_id": subject_id_oid})
            subject_name = (
                subject_doc.get("name", "Unknown Subject")
                if subject_doc
                else "Unknown Subject"
            )

            collection = self.mongo.get_collection("studenttopicprogresses")

            # Find all entries for this subject and user
            cursor = collection.find(
                {"subjectId": subject_id_oid, "userId": user_id_oid}
            )
            progress_entries = list(cursor)

            if not progress_entries:
                return "No progress records found for this Subject ID."

            # Collect IDs for batch fetching
            chapter_ids = set()
            topic_ids = set()
            for entry in progress_entries:
                if "chapterId" in entry:
                    chapter_ids.add(entry["chapterId"])
                if "topicId" in entry:
                    topic_ids.add(entry["topicId"])

            # Fetch Chapter Names
            chapters_coll = self.mongo.get_collection("chapters")
            chapters_cursor = chapters_coll.find({"_id": {"$in": list(chapter_ids)}})
            chapter_map = {
                str(doc["_id"]): doc.get("name", "Unknown Chapter")
                for doc in chapters_cursor
            }

            # Fetch Topic Titles
            topics_coll = self.mongo.get_collection("topics")
            topics_cursor = topics_coll.find({"_id": {"$in": list(topic_ids)}})
            topic_map = {
                str(doc["_id"]): doc.get("title", "Unknown Topic")
                for doc in topics_cursor
            }

            # Aggregate data by chapter and topic
            chapters_data = {}

            for entry in progress_entries:
                chapter_id_oid = entry.get("chapterId")
                chapter_id = str(chapter_id_oid) if chapter_id_oid else "unknown"

                topic_id_oid = entry.get("topicId")
                topic_id = str(topic_id_oid) if topic_id_oid else "unknown"

                # Clean up entry for LLM
                clean_entry = {
                    "topicId": topic_id,
                    "topicTitle": topic_map.get(topic_id, "Unknown Topic"),
                    "totalAttempts": entry.get("totalAttempts", 0),
                    "easyAttempts": entry.get("easyAttempts", 0),
                    "mediumAttempts": entry.get("mediumAttempts", 0),
                    "hardAttempts": entry.get("hardAttempts", 0),
                    "easyCorrect": entry.get("easyCorrect", 0),
                    "mediumCorrect": entry.get("mediumCorrect", 0),
                    "hardCorrect": entry.get("hardCorrect", 0),
                    "avgTimeSpent": entry.get("avgTimeSpent", 0),
                    "masteryScore": entry.get("masteryScore", 0),
                    "masteryState": entry.get("masteryState", "UNKNOWN"),
                }

                if chapter_id not in chapters_data:
                    chapters_data[chapter_id] = {
                        "name": chapter_map.get(chapter_id, "Unknown Chapter"),
                        "topics": [],
                    }

                chapters_data[chapter_id]["topics"].append(clean_entry)

            # Prepare prompt for LLM
            prompt_data = []
            prompt_data.append(f"Subject: {subject_name}\n")

            for chap_id, data in chapters_data.items():
                prompt_data.append(f"Chapter: {data['name']} (ID: {chap_id})")
                for topic in data["topics"]:
                    prompt_data.append(
                        f"  - Topic: {topic['topicTitle']} (ID: {topic['topicId']})"
                    )
                    prompt_data.append(
                        f"    Mastery: {topic['masteryState']} (Score: {topic['masteryScore']:.2f})"
                    )
                    prompt_data.append(
                        f"    Attempts: {topic['totalAttempts']} (Easy: {topic['easyAttempts']}, Medium: {topic['mediumAttempts']}, Hard: {topic['hardAttempts']})"
                    )
                    prompt_data.append(
                        f"    Correct: Easy: {topic['easyCorrect']}, Medium: {topic['mediumCorrect']}, Hard: {topic['hardCorrect']}"
                    )
                prompt_data.append("")

            prompt_text = "\n".join(prompt_data)

            system_prompt = (
                "You are an friendly expert educational AI assistant. "
                "Analyze the following student progress data grouped by chapters. "
                "Identify which chapters are weak and need more practice. "
                "Provide only chapter based analysis do not provide topic based analysis."
                "Output should be in short paragraph 40 to 50 words only"
            )

            user_prompt = f"Here is the student's progress data for subject '{subject_name}':\n\n{prompt_text}\n\nPlease analyze this and tell me which chapters are weak and need practice."

            logger.info("Sending data to Gemini LLM for analysis...")

            # Get response from LLM
            response_generator = self.llm.get_response(
                system_prompt=system_prompt, user_prompt=user_prompt
            )

            full_response = ""
            for chunk in response_generator:
                if isinstance(chunk, dict) and "response" in chunk:
                    text = chunk["response"]
                    full_response += text

            return full_response

        except Exception as e:
            logger.exception(f"An error occurred during subject analysis: {e}")
            return f"An error occurred: {str(e)}"

    def get_ai_insights_chapter(self, chapter_id: str, user_id: str) -> str:
        try:
            try:
                chapter_id_oid = ObjectId(chapter_id)
                user_id_oid = ObjectId(user_id)
            except Exception as e:
                return f"Invalid ID format: {e}"

            logger.info(
                f"Fetching analysis for Chapter ID: {chapter_id_oid} and User ID: {user_id_oid}..."
            )

            # Fetch Chapter Name
            chapters_coll = self.mongo.get_collection("chapters")
            chapter_doc = chapters_coll.find_one({"_id": chapter_id_oid})

            if not chapter_doc:
                return "Chapter not found."

            chapter_name = chapter_doc.get("name", "Unknown Chapter")

            # 1. Get all topics expected in this chapter
            chapter_topics_coll = self.mongo.get_collection("chaptertopics")
            all_topics_cursor = chapter_topics_coll.find({"chapterId": chapter_id_oid})

            all_topic_ids = []
            for doc in all_topics_cursor:
                if "topicId" in doc:
                    all_topic_ids.append(doc["topicId"])

            # 2. Get student progress for this chapter
            progress_coll = self.mongo.get_collection("studenttopicprogresses")
            progress_cursor = progress_coll.find(
                {"chapterId": chapter_id_oid, "userId": user_id_oid}
            )
            progress_entries = list(progress_cursor)

            attempted_topic_ids_set = set()
            topic_progress_map = {}  # topicId (str) -> progress entry

            for entry in progress_entries:
                if "topicId" in entry:
                    tid = entry["topicId"]
                    attempted_topic_ids_set.add(tid)
                    topic_progress_map[str(tid)] = entry

            # 3. Identify uncovered topics
            uncovered_topic_ids = []
            for tid in all_topic_ids:
                if tid not in attempted_topic_ids_set:
                    uncovered_topic_ids.append(tid)

            # 4. Fetch Names for ALL topics (both covered and uncovered)
            all_involved_topic_ids = set(all_topic_ids) | attempted_topic_ids_set

            topics_coll = self.mongo.get_collection("topics")
            topics_cursor = topics_coll.find(
                {"_id": {"$in": list(all_involved_topic_ids)}}
            )
            topic_name_map = {
                str(doc["_id"]): doc.get("title", "Unknown Topic")
                for doc in topics_cursor
            }

            # 5. Prepare Prompt Data
            prompt_data = []
            prompt_data.append(f"Chapter: {chapter_name}\n")

            # Section A: Covered Topics Progress
            prompt_data.append("Covered Topics Progress:")
            if not attempted_topic_ids_set:
                prompt_data.append("  (None)")
            else:
                for tid_oid in attempted_topic_ids_set:
                    tid_str = str(tid_oid)
                    entry = topic_progress_map.get(tid_str)
                    t_name = topic_name_map.get(tid_str, "Unknown Topic")
                    if entry:
                        prompt_data.append(f"  - Topic: {t_name}")
                        prompt_data.append(
                            f"    Mastery: {entry.get('masteryState', 'UNKNOWN')} (Score: {entry.get('masteryScore', 0):.2f})"
                        )
                        prompt_data.append(
                            f"    Attempts: {entry.get('totalAttempts', 0)}"
                        )

            prompt_data.append("\nUncovered Topics (Not yet attempted):")
            if not uncovered_topic_ids:
                prompt_data.append("  (None - All topics attempted)")
            else:
                for tid_oid in uncovered_topic_ids:
                    tid_str = str(tid_oid)
                    t_name = topic_name_map.get(tid_str, "Unknown Topic")
                    prompt_data.append(f"  - {t_name}")

            prompt_text = "\n".join(prompt_data)

            system_prompt = (
                "You are an friendly expert educational AI assistant. "
                "Analyze the following progress data for a specific chapter. "
                "1. Evaluate performance on the covered topics. "
                "2. Identify specific topics that have NOT been attempted yet and encourage the student to start them. "
                "Refer to topics by name. "
                "Output should be a concise paragraph (approx 50 words)."
            )

            user_prompt = f"Here is the progress data for chapter '{chapter_name}':\n\n{prompt_text}\n\nPlease analyze the progress and missing topics."

            logger.info("Sending data to Gemini LLM for analysis...")

            # Get response from LLM
            response_generator = self.llm.get_response(
                system_prompt=system_prompt, user_prompt=user_prompt
            )

            full_response = ""
            for chunk in response_generator:
                if isinstance(chunk, dict) and "response" in chunk:
                    text = chunk["response"]
                    full_response += text

            return full_response

        except Exception as e:
            logger.exception(f"An error occurred during chapter analysis: {e}")
            return f"An error occurred: {str(e)}"
