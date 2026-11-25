@echo off
REM ============================================================
REM Startup script for the Movie Recommendation Website
REM Sets env vars BEFORE Python starts to fix Windows DLL conflicts
REM ============================================================
set KMP_DUPLICATE_LIB_OK=TRUE
set TOKENIZERS_PARALLELISM=false
cd /d "C:\Users\abhin\OneDrive\Desktop\movies\backend"
python app.py
pause
