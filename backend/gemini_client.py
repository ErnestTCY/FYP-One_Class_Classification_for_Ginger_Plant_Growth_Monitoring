import os, requests

def gemini_reasoning(summary_text: str) -> str:
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        # fallback: echo-friendly stub
        return "Gemini not configured. Draft summary:\n" + summary_text
    # Minimal REST call (pseudo; replace with the actual Gemini endpoint you use)
    url = "https://generativelanguage.googleapis.com/v1/models/gemini-pro:generateContent"
    headers = {"Content-Type": "application/json"}
    payload = {"contents":[{"parts":[{"text": summary_text}]}]}
    r = requests.post(f"{url}?key={api_key}", headers=headers, json=payload, timeout=60)
    r.raise_for_status()
    data = r.json()
    # Extract text safely:
    return data.get("candidates",[{}])[0].get("content",{}).get("parts",[{}])[0].get("text","(no text)")
