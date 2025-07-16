import os
import argparse
from dotenv import load_dotenv
import google.generativeai as genai
from typing import List

# Carga de variables de entorno y configuración de la API
def load_api_key():
    load_dotenv()
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("Falta GEMINI_API_KEY en el entorno.")
    genai.configure(api_key=api_key)

MODEL_NAME = "gemini-1.5-flash"

def call_gemini_analyzer(text: str) -> List[str]:
    """
    Llama a la API de Gemini para generar un resumen de un CSV.
    Args:
        text (str): Texto en formato CSV.
    Returns:
        List[str]: Lista de líneas del resumen.
    """
    prompt = (
        "Eres un asistente que resume datos presentados en CSV. "
        "El siguiente bloque de texto está en formato CSV, con comas como separador y comillas para campos multilínea. "
        "Genera un resumen de los siguientes datos.\n\n"
        + text
    )
    try:
        model = genai.GenerativeModel(MODEL_NAME)
        response = model.generate_content([prompt])
        return response.text.splitlines()
    except Exception as e:
        print(f"Error con Gemini: {e}")
        return []
