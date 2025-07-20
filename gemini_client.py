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
    # Llama a la API de Gemini para extraer ideas principales del texto.
    prompt = (
        "Eres un asistente experto en comprensión de textos. "
        "Extrae entre 5 y 8 ideas principales del siguiente texto. "
        "Cada idea debe cubrir un concepto único y relevante. "
        "No repitas ideas ni reformules lo mismo. "
        "Incluye términos técnicos importantes si aparecen. "
        "Escribe cada idea como una o varias oraciones, y numéralas así: 1., 2., 3., etc.\n\n"
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
    #busca las líneas que comienzan con un número seguido de punto y espacio
    for line in content.splitlines():
        #representan las ideas principales
        m = re.match(r"^\s*\d+\.\s*(.+)", line)
        #captura el texto de la idea, lo limpia de espacios innecesarios y lo agrega a la lista de ideas si no está repetido
        if m:
            idea = m.group(1).strip()
            if idea and idea not in ideas:
                ideas.append(idea)
    # Retorna hasta 8 ideas únicas
    return ideas[:8]