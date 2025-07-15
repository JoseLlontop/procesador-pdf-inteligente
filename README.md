# Procesador de PDF Inteligente

Este proyecto es una aplicación web interactiva creada con Streamlit que permite procesar archivos PDF, limpiar el texto extraído y generar ideas principales, preguntas y respuestas utilizando servicios externos (Gemini y DeepSeek).

## Ejecución Local

1. **Clonar repositorio**
   ```bash
   git clone <URL_DEL_REPOSITORIO>
   cd <NOMBRE_DEL_PROYECTO>
   ```

2. **Crear y activar entorno virtual (opcional pero recomendado)**
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # En Windows: venv\Scripts\activate
   ```

3. **Instalar dependencias**
   ```bash
   pip install -r requirements.txt
   ```

4. **Configurar variables de entorno**
   - Crear un archivo `.env` en la raíz del proyecto con tus claves de API:
     ```
     GEMINI_API_KEY=tu_clave_gemini
     DEEPSEEK_API_KEY=tu_clave_deepseek
     ```
   - Asegúrate de no subir el archivo `.env` al repositorio (agregar a `.gitignore`).

5. **Ejecutar la aplicación**
   ```bash
   streamlit run app.py
   ```
   - Abre tu navegador en `http://localhost:8501`

## Ejecución con Docker

1. **Construir la imagen Docker**
   ```bash
   docker build -t procesador-pdf-inteligente .
   ```

2. **Ejecutar el contenedor**
   ```bash
   docker run -d -p 8501:8501 --name pdf-inteligente      --env-file .env      procesador-pdf-inteligente
   ```
   - `--env-file .env`: carga tus variables de entorno desde el archivo `.env`.

3. **Acceder a la aplicación**
   - Abre tu navegador en `http://localhost:8501`
