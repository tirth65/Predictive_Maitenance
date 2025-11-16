@echo off
echo Starting Predictive Maintenance System...
echo.

echo Starting ML Service (Flask)...
start "ML Service" cmd /k "cd ml-model & python app.py"

echo Waiting 3 seconds...
timeout /t 3 /nobreak >nul

echo Starting Backend Service (Node.js)...
start "Backend Service" cmd /k "cd backend & npm start"

echo Waiting 3 seconds...
timeout /t 3 /nobreak >nul

echo Starting Frontend (React)...
start "Frontend Service" cmd /k "cd frontend & npm run dev"

echo.
echo All services are starting...
echo.
echo ML Service: http://localhost:8000
echo Backend: http://localhost:5000
echo Frontend: http://localhost:5173
echo.
echo Press any key to test connection...
pause >nul

echo Testing connection...
python test_connection.py

echo.
echo Press any key to exit...
pause >nul
