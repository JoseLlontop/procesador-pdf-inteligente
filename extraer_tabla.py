import pdfplumber
import csv
import io
from pathlib import Path
import logging
from limpieza_texto import clean_text 

logger = logging.getLogger(__name__)

def extraer_tablas(path_pdf: str = "texto_ia.pdf") -> str:
    """
    Abre un PDF, extrae tablas por página, limpia cada celda y devuelve el resultado en formato CSV como string.
    Devuelve cadena vacía si no encuentra tablas o hay error.
    """
    try:
        pdf_file = Path(path_pdf)
        if not pdf_file.is_file():
            logger.error(f"El archivo no existe: {path_pdf}")
            return ""
        
        buffer = io.StringIO()
        csv_writer = csv.writer(buffer)
        tablas_encontradas = False
        
        with pdfplumber.open(path_pdf) as pdf:
            for page in pdf.pages:
                tablas = page.extract_tables()
                if not tablas:
                    continue
                
                for tabla in tablas:
                    tablas_encontradas = True
                    for fila in tabla:
                        fila_limpia = [
                            clean_text(celda) if celda is not None else "" 
                            for celda in fila
                        ]
                        csv_writer.writerow(fila_limpia)
                    buffer.write("\n")
        
        return buffer.getvalue() if tablas_encontradas else ""
    
    except Exception as e:
        logger.error(f"Error extrayendo tablas: {str(e)}")
        return ""