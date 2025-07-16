import os
import json
import re
from openai import OpenAI
# Carga de API key
from dotenv import load_dotenv
load_dotenv()

API_KEY = os.getenv("DEEPSEEK_API_KEY")
if not API_KEY:
    raise ValueError("Falta DEEPSEEK_API_KEY en el entorno")

client = OpenAI(api_key=API_KEY, base_url="https://api.deepseek.com")

def call_deepseek(ideas: list[str], num_questions: int = 8) -> list[dict]:
    """
    Genera preguntas MCQ en español basadas en 'ideas'.
    Devuelve lista de dicts con 'question', 'options' y 'correct_answer'.
    """
    system_msg = {
        "role": "system",
        "content": (
            f"Eres un experto diseñador de evaluaciones académicas en español."
            f"Genera EXACTAMENTE {num_questions} preguntas de opción múltiple claras, variadas y relevantes"
            "Incluye 4 opciones por pregunta y marca la correcta. "
            "Cada pregunta debe abordar un concepto importante distinto del texto, "
            "evitando repeticiones temáticas y cubriendo la mayor cantidad posible de ideas. "
            "Las preguntas deben ser precisas, no ambiguas, y con distractores plausibles pero incorrectos. "
            "IMPORTANTE: Devuelve ÚNICA y EXCLUSIVAMENTE un JSON válido con clave \"questions\" "
            "y dentro de cada elemento: \"question\", \"options\" y \"correct_answer\"."
        )
    }
    user_msg = {
        "role": "user",
        "content": json.dumps(
            {"ideas": ideas, "num_questions": num_questions},
            ensure_ascii=False
        )
    }

    try:
        resp = client.chat.completions.create(
            model="deepseek-chat",
            messages=[system_msg, user_msg],
            stream=False
        )
        raw = resp.choices[0].message.content
        # Extraer JSON aunque venga en un fence ```json
        m = re.search(r"```(?:json)?\n([\s\S]+?)```", raw)
        payload = m.group(1).strip() if m else raw.strip()
        data = json.loads(payload)
        return data.get("questions", [])
    
    except Exception as e:
        print(f"[ERROR] DeepSeek: {e}")
        return []