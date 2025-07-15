from PyPDF2 import PdfReader

def extract_text_from_pdf(path: str) -> str:
    """
    Lee un PDF y devuelve todo su texto concatenado.
    """
    reader = PdfReader(path)
    text = []
    for page in reader.pages:
        text.append(page.extract_text() or "")
    return "\n".join(text)