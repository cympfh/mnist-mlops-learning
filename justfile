# uvicorn backend
backend:
    uvicorn --host 0.0.0.0 --reload --reload-dir backend backend.main:app

# streamlit frontend
frontend:
    streamlit run frontend/main.py

# mlflow ui
mlflow:
    mlflow ui --backend-store-uri sqlite:///db/backend.db
