@echo off
chcp 65001 >nul
cd /d "%~dp0"
title smartMonitoring - 运行记录（关掉这个窗口不会停止机器人）
if not exist data\bot_hf.log (
  echo 还没有记录，先双击 start_bot.bat 启动机器人。
  pause
  exit /b
)
powershell -NoProfile -Command "Get-Content data\bot_hf.log -Tail 40 -Wait -Encoding UTF8"
