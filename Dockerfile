# 1. Usa una imagen base de Python ligera
FROM python:3.10-slim

# 2. Establece el directorio de trabajo
WORKDIR /app

# 3. Copia los archivos de requerimientos e instala dependencias
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# 4. Copia el resto de tu código
COPY . .

# 5. Exponer el puerto que usa Streamlit
EXPOSE 8501

# 6. Comando por defecto para lanzar la app
ENTRYPOINT ["streamlit", "run"]
CMD ["app.py", "--server.port=8501", "--server.address=0.0.0.0"]