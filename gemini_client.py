import os
import re
import google.generativeai as genai
from typing import List
from dotenv import load_dotenv

load_dotenv()
API_KEY = os.getenv("GEMINI_API_KEY")
genai.configure(api_key=API_KEY)
MODEL_NAME = "gemini-1.5-flash"

def call_gemini(text: str) -> List[str]:
    prompt = (
        "Eres un asistente experto en análisis de texto. "
        "Extrae entre 3 y 5 ideas principales del texto proporcionado. "
        "Cada idea puede ocupar varias oraciones. "
        "Enumera cada idea con numeración (1., 2., etc.) al inicio de cada línea.\n\n"
        + text
    )
    try:
        model = genai.GenerativeModel(MODEL_NAME)
        response = model.generate_content([prompt])
        content = response.text
    except Exception as e:
        print(f"Error con Gemini: {e}")
        return []

    # Parseo de ideas numeradas
    ideas = []
    for line in content.splitlines():
        m = re.match(r"^\s*\d+\.\s*(.+)", line)
        if m:
            idea = m.group(1).strip()
            if idea and idea not in ideas:
                ideas.append(idea)
    return ideas[:5]