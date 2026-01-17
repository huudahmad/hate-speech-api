FROM python:3.9-slim
WORKDIR /app

COPY requirements.txt .

#dont need cache for pip installs in docker
#dont need GPU version of torch in AWS (more expensive, not needed for inference)
#need to set pip install priority so it doesnt try to install heavier GPU version
RUN pip install --no-cache-dir -r requirements.txt \
    --index-url https://download.pytorch.org/whl/cpu \
    --extra-index-url https://pypi.org/simple

COPY app/ ./app
COPY hate_speech_model/ ./hate_speech_model

#start the container 
#we use "uvicorn" (the server) to load "app" (the FastAPI logic) from "app/main.py".
#Host 0.0.0.0 exposes the container to the outside world (laptop/AWS).
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]