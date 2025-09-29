import os
import requests

def gemini_reasoning(summary_text: str) -> str:
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        return "Gemini not configured. Draft summary:\n" + summary_text
    
    url = "https://generativelanguage.googleapis.com/v1/models/gemini-2.5-flash:generateContent"
    headers = {"Content-Type": "application/json"}
    payload = {"contents":[{"parts":[{"text": summary_text}]}]}
    r = requests.post(f"{url}?key={api_key}", headers=headers, json=payload, timeout=60)
    r.raise_for_status()
    data = r.json()
    return data.get("candidates",[{}])[0].get("content",{}).get("parts",[{}])[0].get("text","(no text)")
