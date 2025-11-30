import requests
import os

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_URL = os.getenv("OPENROUTER_URL")
MODEL = os.getenv("MODEL")

def call_llm(system_prompt: str, user_prompt: str, temperature: float = 0.7) -> str:
    """
    Minimal LLM caller extracted from your QuestionService.
    Returns the raw text output from the model.
    """

    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
    }

    payload = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": temperature,
    }

    response = requests.post(OPENROUTER_URL, headers=headers, json=payload, timeout=30)

    if not response.ok:
        raise RuntimeError(f"LLM API Error {response.status_code}: {response.text}")

    data = response.json()
    return data["choices"][0]["message"]["content"]
