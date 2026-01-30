@echo off

REM 安装依赖
pip install -r requirements-web.txt

REM 启动应用
python webapp.py

pause