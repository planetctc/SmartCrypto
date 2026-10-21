@echo off
chcp 65001 >nul
cd /d "%~dp0"
set PIDFILE=data\state_hf.pid
if not exist "%PIDFILE%" (
  echo 机器人没有在运行。
  timeout /t 5 >nul
  exit /b
)
set /p OLDPID=<"%PIDFILE%"
taskkill /PID %OLDPID% /F >nul 2>&1
del "%PIDFILE%" >nul 2>&1
echo 机器人已停止。已经开着的单在 Bitget 上仍有止损和止盈保护。
timeout /t 5 >nul
