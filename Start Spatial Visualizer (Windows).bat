@echo off
rem Double-click to start the Spatial Transcriptomics Visualizer (Windows).
rem
rem If the GitHub repository is reachable (i.e. public), each launch runs the latest
rem version from it. Otherwise (private repo, or no internet) it runs the copy of the app
rem in this folder. Either way uv provides a private Python + packages - nothing is
rem installed system-wide and no admin rights are needed.

setlocal
title Spatial Transcriptomics Visualizer

set "PYTHON_VERSION=3.13"

set "HERE=%~dp0"
if "%HERE:~-1%"=="\" set "HERE=%HERE:~0,-1%"
set "PATH=%USERPROFILE%\.local\bin;%USERPROFILE%\.cargo\bin;%PATH%"
rem Saved colors/fields stay in this folder whether the online or the local copy runs.
set "SPATIAL_VISUALIZER_CACHE_DIR=%HERE%\cache"

where uv >nul 2>nul
if errorlevel 1 (
    echo First-time setup: installing uv ^(the tool that runs the app^)...
    powershell -NoProfile -ExecutionPolicy ByPass -Command "irm https://astral.sh/uv/install.ps1 | iex"
    if errorlevel 1 goto :fail
)

echo Starting the Spatial Transcriptomics Visualizer.
echo The first launch downloads everything it needs and can take a few minutes.
echo The app opens in your browser. Closing the browser tab stops it.
echo.

echo Using the version in this folder.
uv run --managed-python --python %PYTHON_VERSION% --project "%HERE%" spatial-visualizer
if errorlevel 1 goto :fail
goto :eof

:fail
echo.
echo Something went wrong ^(see the messages above^).
pause
exit /b 1
