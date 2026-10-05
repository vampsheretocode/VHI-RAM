#!/bin/bash
# Start FastAPI backend in the background
python -m uvicorn src.blend.api.app:app --host 0.0.0.0 --port 8000 &

# Start Streamlit frontend in the foreground
python -m streamlit run src/blend/dashboard/app.py --server.port 8501 --server.address 0.0.0.0
