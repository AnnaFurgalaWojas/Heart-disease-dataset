# Używamy lekkiego obrazu Pythona
FROM python:3.11-slim

# Ustaw katalog roboczy
WORKDIR /app

# Skopiuj plik z wymaganiami
COPY requirements.txt .

# Zainstaluj zależności
RUN pip install --no-cache-dir -r requirements.txt

# Skopiuj resztę kodu aplikacji
COPY . .

# Otwórz port (FastAPI domyślnie działa na 8000)
EXPOSE 8000

# Komenda startowa (uvicorn uruchamia FastAPI)
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]