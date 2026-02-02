# Build Windows bundle (onedir) with PyInstaller
$ErrorActionPreference = "Stop"
$python = "C:\Users\hwang\miniconda3\envs\yolo\python.exe"
$project = "C:\Users\hwang\microscopy-yolov8"
Set-Location $project

& $python -m pip install --upgrade pyinstaller

# Clean previous builds
if (Test-Path "$project\build") { Remove-Item -Recurse -Force "$project\build" }
if (Test-Path "$project\dist") { Remove-Item -Recurse -Force "$project\dist" }

& $python -m PyInstaller --noconfirm --onedir --windowed --name MicroscopyAI `
  --add-data "ui;ui" `
  --add-data "core;core" `
  --add-data "data;data" `
  --add-data "docs;docs" `
  --add-data "third_party;third_party" `
  main.py

Write-Host "Build complete. Output: $project\dist\MicroscopyAI"
