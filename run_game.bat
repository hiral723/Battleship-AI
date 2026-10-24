@echo off
REM Double-click to install pygame (first time) and start the game.
where py >nul 2>nul && (set PY=py) || (set PY=python)
%PY% -m pip install -r requirements.txt
%PY% main.py
pause
