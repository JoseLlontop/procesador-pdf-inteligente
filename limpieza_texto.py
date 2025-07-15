import re

def clean_text(text: str) -> str:
    """
    Limpia el texto eliminando saltos excesivos,
    caracteres no ASCII y espacios repetidos.
    """
    cleaned = re.sub(r"\n{2,}", "\n", text)
    cleaned = re.sub(r"[^\x00-\x7F\n\.,;:¿?¡!%()\-_\s]+", "", cleaned)
    cleaned = re.sub(r"[ ]{2,}", " ", cleaned)
    return cleaned.strip()