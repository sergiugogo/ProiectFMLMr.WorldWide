#!/usr/bin/env bash
# Run the Streamlit UI (port set in .streamlit/config.toml → 8001)
# Backend must be running first: uvicorn api:app --port 8000
cd "$(dirname "$0")"
exec streamlit run app.py
