# Run OptiRND FastAPI Server
Write-Host "Starting FastAPI server on http://127.0.0.1:8001" -ForegroundColor Green

python -m uvicorn api.server:app --host 127.0.0.1 --port 8001 --reload