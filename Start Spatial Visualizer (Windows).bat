@echo off
rem Double-click to start the Spatial Transcriptomics Visualizer (Windows).
rem
rem Runs the copy of the app in this folder. uv provides a private Python + packages -
rem nothing is installed system-wide and no admin rights are needed. --frozen installs
rem exactly the versions in uv.lock, so every computer gets the same environment.

setlocal
title Spatial Transcriptomics Visualizer

set "PYTHON_VERSION=3.13"

set "HERE=%~dp0"
if "%HERE:~-1%"=="\" set "HERE=%HERE:~0,-1%"
set "PATH=%USERPROFILE%\.local\bin;%USERPROFILE%\.cargo\bin;%PATH%"
rem Saved colors/fields/regions are kept in this folder's cache\.
set "SPATIAL_VISUALIZER_CACHE_DIR=%HERE%\cache"

where uv >nul 2>nul
if errorlevel 1 (
    echo First-time setup: installing uv ^(the tool that runs the app^)...
    powershell -NoProfile -ExecutionPolicy ByPass -Command "irm https://astral.sh/uv/install.ps1 | iex"
    if errorlevel 1 goto :fail
)

echo Starting the Spatial Transcriptomics Visualizer.
echo The first launch downloads everything it needs and can take a few minutes.
echo The app opens in your browser. Closing the browser tab stops it ^(after about 30 s^).
echo.

echo Using the version in this folder.
uv run --frozen --managed-python --python %PYTHON_VERSION% --project "%HERE%" spatial-visualizer
if errorlevel 1 goto :fail
goto :eof

:fail
echo.
echo Something went wrong ^(see the messages above^).
pause
exit /b 1
