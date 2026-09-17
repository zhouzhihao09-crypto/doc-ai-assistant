@echo on

cd /d "C:\Users\zhiha\OneDrive\Desktop\real world problems\doc_ai_assistant"

echo.
echo ===== Starting AI Website =====
echo Current folder:
cd

echo.
echo ===== Python =====
".venv\Scripts\python.exe" --version

echo.
echo ===== Starting Uvicorn =====
".venv\Scripts\python.exe" -m uvicorn app:app --host 0.0.0.0 --port 8000

echo.
echo ===== Server stopped =====
pause