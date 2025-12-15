@echo off
setlocal

rmdir /s /q build 2>nul
rmdir /s /q dist 2>nul

pyinstaller ^
  --noconfirm ^
  --clean ^
  --name PointAI ^
  --windowed ^
  --onedir ^
  --hidden-import pyqtgraph.opengl ^
  --hidden-import OpenGL ^
  --hidden-import zarr ^
  --hidden-import numcodecs ^
  main.py

echo DONE. Check dist\PointAI\
endlocal
