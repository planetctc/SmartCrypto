@echo off
chcp 65001 >nul
cd /d "%~dp0"
set CFG=config_hf.yaml
set PIDFILE=data\state_hf.pid
if not exist "%PIDFILE%" goto checkkey
set /p OLDPID=<"%PIDFILE%"
tasklist /FI "PID eq %OLDPID%" /FI "IMAGENAME eq pythonw.exe" 2>nul | find "%OLDPID%" >nul
if errorlevel 1 goto checkkey
echo 机器人已经在运行了，不用再启动。
echo 想看它在做什么，双击 view_log.bat
timeout /t 5 >nul
exit /b

:checkkey
if not "%BITGET_API_KEY%"=="" goto run
echo 还没有设置 API key，请先双击 set_api_key.bat
pause
exit /b

:run
start "" pyw -3.13 main.py run --config %CFG%
echo smartMonitoring 已在后台启动，没有窗口。
echo 想看它在做什么，双击 view_log.bat
echo 要停止，双击 stop_bot.bat
timeout /t 5 >nul
