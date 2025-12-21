import os
import sys
from dotenv import load_dotenv
from bson import ObjectId

# Add project root to python path to allow imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from db.mongo_db import MongoDB
from llm.llm_open_router import LLMService


def analyze_student_progress_subject(subject_id):
    # Load environment variables
    load_dotenv()

    # Initialize services
    try:
        mongo = MongoDB(os.getenv("MONGO_URI"), os.getenv("MONGO_DB_NAME"))
        llm = LLMService()
    except Exception as e:
        print(f"Error initializing services: {e}")
        return

    try:
        subject_id = ObjectId(subject_id)
    except Exception as e:
        print(f"Invalid Subject ID format: {e}")
        return

    print(f"\nFetching progress for Subject ID: {subject_id}...")

    try:
        # Fetch Subject Name
        subjects_coll = mongo.get_collection("subjects")
        subject_doc = subjects_coll.find_one({"_id": subject_id})
        subject_name = (
            subject_doc.get("name", "Unknown Subject")
            if subject_doc
            else "Unknown Subject"
        )
        print(f"Subject: {subject_name}")

        collection = mongo.get_collection("studenttopicprogresses")

        # Find all entries for this subject
        cursor = collection.find({"subjectId": subject_id})
        progress_entries = list(cursor)

        if not progress_entries:
            print("No progress records found for this Subject ID.")
            return

        print(
            f"Found {len(progress_entries)} progress records. Fetching metadata and aggregating data..."
        )

        # Collect IDs for batch fetching
        chapter_ids = set()
        topic_ids = set()
        for entry in progress_entries:
            if "chapterId" in entry:
                chapter_ids.add(entry["chapterId"])
            if "topicId" in entry:
                topic_ids.add(entry["topicId"])

        # Fetch Chapter Names
        chapters_coll = mongo.get_collection("chapters")
        chapters_cursor = chapters_coll.find({"_id": {"$in": list(chapter_ids)}})
        chapter_map = {
            str(doc["_id"]): doc.get("name", "Unknown Chapter")
            for doc in chapters_cursor
        }

        # Fetch Topic Titles
        topics_coll = mongo.get_collection("topics")
        topics_cursor = topics_coll.find({"_id": {"$in": list(topic_ids)}})
        topic_map = {
            str(doc["_id"]): doc.get("title", "Unknown Topic") for doc in topics_cursor
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

        print("\nSending data to Gemini LLM for analysis...\n")

        # Get response from LLM
        response_generator = llm.get_response(
            system_prompt=system_prompt, user_prompt=user_prompt
        )

        full_response = ""
        print("--- LLM Insights ---")
        for chunk in response_generator:
            if isinstance(chunk, dict) and "response" in chunk:
                text = chunk["response"]
                print(text, end="", flush=True)
                full_response += text
        print("\n--------------------")

    except Exception as e:
        print(f"An error occurred during analysis: {e}")
    finally:
        mongo.close()


def analyze_student_progress_chapter(chapter_id):
    # Load environment variables
    load_dotenv()

    # Initialize services
    try:
        mongo = MongoDB(os.getenv("MONGO_URI"), os.getenv("MONGO_DB_NAME"))
        llm = LLMService()
    except Exception as e:
        print(f"Error initializing services: {e}")
        return

    try:
        chapter_id_oid = ObjectId(chapter_id)
    except Exception as e:
        print(f"Invalid Chapter ID format: {e}")
        return

    print(f"\nFetching analysis for Chapter ID: {chapter_id_oid}...")

    try:
        # Fetch Chapter Name
        chapters_coll = mongo.get_collection("chapters")
        chapter_doc = chapters_coll.find_one({"_id": chapter_id_oid})

        if not chapter_doc:
            print("Chapter not found.")
            return

        chapter_name = chapter_doc.get("name", "Unknown Chapter")
        print(f"Chapter: {chapter_name}")

        # 1. Get all topics expected in this chapter
        chapter_topics_coll = mongo.get_collection("chaptertopics")
        all_topics_cursor = chapter_topics_coll.find({"chapterId": chapter_id_oid})

        all_topic_ids = []
        for doc in all_topics_cursor:
            if "topicId" in doc:
                all_topic_ids.append(doc["topicId"])

        if not all_topic_ids:
            print("No topics found for this chapter in 'chaptertopics' collection.")
            # Depending on logic, we might still check progress, but usually this implies empty chapter structure

        print(f"Total topics in chapter: {len(all_topic_ids)}")

        # 2. Get student progress for this chapter
        progress_coll = mongo.get_collection("studenttopicprogresses")
        # Assuming we just want *any* progress for this chapter, regardless of user?
        # The original request implied "student progress", usually for a specific user.
        # However, the previous function took subject_id and found ALL entries (which might be mixed users if not filtered).
        # The prompt says "fetch all entries of given subject id... Since for one subject id multiple entry with same chapter id can exist".
        # This implies we are aggregating across users OR the DB only has one user's data?
        # The schema has `userId`.
        # The previous function `analyze_student_progress_subject` did NOT filter by userId.
        # I will follow similar logic: fetch based on chapterId.

        progress_cursor = progress_coll.find({"chapterId": chapter_id_oid})
        progress_entries = list(progress_cursor)  # List of dicts

        attempted_topic_ids_set = set()
        topic_progress_map = {}  # topicId (str) -> progress entry

        for entry in progress_entries:
            if "topicId" in entry:
                tid = entry["topicId"]
                attempted_topic_ids_set.add(tid)
                topic_progress_map[str(tid)] = entry

        print(f"Attempted topics: {len(attempted_topic_ids_set)}")

        # 3. Identify uncovered topics
        uncovered_topic_ids = []
        for tid in all_topic_ids:
            if tid not in attempted_topic_ids_set:
                uncovered_topic_ids.append(tid)

        print(f"Uncovered topics: {len(uncovered_topic_ids)}")

        # 4. Fetch Names for ALL topics (both covered and uncovered)
        # Combine sets
        all_involved_topic_ids = set(all_topic_ids) | attempted_topic_ids_set

        topics_coll = mongo.get_collection("topics")
        topics_cursor = topics_coll.find({"_id": {"$in": list(all_involved_topic_ids)}})
        topic_name_map = {
            str(doc["_id"]): doc.get("title", "Unknown Topic") for doc in topics_cursor
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
                    prompt_data.append(f"    Attempts: {entry.get('totalAttempts', 0)}")

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

        print("\nSending data to Gemini LLM for analysis...\n")

        # Get response from LLM
        response_generator = llm.get_response(
            system_prompt=system_prompt, user_prompt=user_prompt
        )

        full_response = ""
        print("--- LLM Insights ---")
        for chunk in response_generator:
            if isinstance(chunk, dict) and "response" in chunk:
                text = chunk["response"]
                print(text, end="", flush=True)
                full_response += text
        print("\n--------------------")

    except Exception as e:
        print(f"An error occurred during chapter analysis: {e}")
    finally:
        mongo.close()


if __name__ == "__main__":
    analyze_student_progress_chapter("686296e6a1abda137f86675d")
    analyze_student_progress_subject("67fed6e64f4451c4718bd135")
