import pdfplumber  # Librería para procesar PDFs
from limpieza_texto import clean_text  # Función de limpieza de texto externalizada
import csv  # Para escribir en formato CSV
import io  # Para buffer en memoria
from pathlib import Path

def extraer_tablas(path_pdf: str = "texto_ia.pdf") -> str:
    """
    Abre un PDF, extrae tablas por página, limpia cada celda y devuelve el resultado en formato CSV como string.

    Args:
        path_pdf (str): Ruta al archivo PDF.

    Returns:
        str: Todo el contenido CSV generado, con tablas separadas por líneas en blanco.
    """
    pdf_file = Path(path_pdf)
    if not pdf_file.is_file():
        raise FileNotFoundError(f"El archivo no existe: {pdf_file}")

    # Buffer en memoria para acumular el CSV
    buffer = io.StringIO()
    csv_writer = csv.writer(buffer)

    with pdfplumber.open(path_pdf) as pdf:
        for page in pdf.pages:
            tablas = page.extract_tables()
            if not tablas:
                continue

            for tabla in tablas:
                for fila in tabla:
                    fila_limpia = [clean_text(celda) if celda is not None else "" for celda in fila]
                    csv_writer.writerow(fila_limpia)
                # Línea en blanco entre tablas
                buffer.write("\n")

    # Obtener el string completo y cerrar buffer
    contenido_csv = buffer.getvalue()
    buffer.close()
    return contenido_csv

