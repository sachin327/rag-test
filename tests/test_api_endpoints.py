import sys
import os
from fastapi.testclient import TestClient

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from main import app

client = TestClient(app)


def test_get_ai_insights_chapter():
    chapter_id = "686296e6a1abda137f86675d"
    print(f"\nTesting GET /ai-insights/chapter/{chapter_id}...")
    response = client.get(f"/ai-insights/chapter/{chapter_id}")
    print(f"Status Code: {response.status_code}")
    if response.status_code == 200:
        print("Response JSON:")
        print(response.json())
    else:
        print("Response Text:")
        print(response.text)


def test_get_ai_insights_subject():
    subject_id = "67fed6e64f4451c4718bd135"
    print(f"\nTesting GET /ai-insights/subject/{subject_id}...")
    response = client.get(f"/ai-insights/subject/{subject_id}")
    print(f"Status Code: {response.status_code}")
    if response.status_code == 200:
        print("Response JSON:")
        print(response.json())
    else:
        print("Response Text:")
        print(response.text)


if __name__ == "__main__":
    test_get_ai_insights_chapter()
    test_get_ai_insights_subject()
