@echo off
chcp 65001 >nul
title SmartCrypto - 设置 API key
echo ============================================
echo   SmartCrypto  设置 Bitget API key
echo   只需要做一次，key 会保存在这台电脑上
echo ============================================
echo.
set /p K=请粘贴 API Key 然后按回车: 
set /p S=请粘贴 Secret Key 然后按回车: 
set /p P=请输入密码短语 Passphrase 然后按回车: 
setx BITGET_API_KEY "%K%" >nul
setx BITGET_API_SECRET "%S%" >nul
setx BITGET_API_PASSPHRASE "%P%" >nul
cls
echo.
echo 已保存。以后直接双击 start_bot.bat 就能启动。
echo.
pause
