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
    Maneja casos vacíos y errores.
    """
    # Si no hay texto para analizar
    if not text.strip():
        return ["No se encontraron tablas para analizar"]
    
    try:
        prompt = (
            "Eres un asistente que resume datos presentados en CSV. "
            "Genera un resumen conciso de máximo 5 puntos clave.\n\n"
            + text
        )
        
        model = genai.GenerativeModel(MODEL_NAME)
        response = model.generate_content([prompt])
        
        if not response.text.strip():
            return ["No se pudo generar interpretación de las tablas"]
            
        return response.text.splitlines()
        
    except Exception as e:
        print(f"Error con Gemini: {e}")
        return [f"Error en interpretación: {str(e)}"]