@echo off
echo Starting MongoDB...
echo.
echo If MongoDB is not installed, you can:
echo 1. Install MongoDB Community Server from https://www.mongodb.com/try/download/community
echo 2. Or run the system without MongoDB (predictions will work but won't be saved)
echo.
echo Starting MongoDB service...
net start MongoDB
if %errorlevel% neq 0 (
    echo.
    echo MongoDB service not found or failed to start.
    echo You can still run the system without MongoDB.
    echo Press any key to continue...
    pause >nul
) else (
    echo MongoDB started successfully!
)
