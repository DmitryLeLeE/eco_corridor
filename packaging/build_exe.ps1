# Builds the Windows application (a folder with ecocorridor.exe) with PyInstaller.
#   pip install . pyinstaller
#   powershell -File packaging/build_exe.ps1
$ErrorActionPreference = "Stop"
pyinstaller --noconfirm --clean --onedir --console --name ecocorridor `
  --collect-data ecocorridor `
  --collect-all rasterio `
  --collect-all pyproj `
  --collect-all pyogrio `
  --collect-all shapely `
  --collect-data geopandas `
  --collect-submodules uvicorn `
  --collect-submodules scipy.sparse.csgraph `
  --hidden-import scipy.special._cdflib `
  packaging/entry.py
Write-Host "Готово: dist/ecocorridor/ecocorridor.exe"
