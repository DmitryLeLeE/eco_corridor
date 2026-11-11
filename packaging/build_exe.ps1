# Builds the Windows application (a folder with ecocorridor.exe) with PyInstaller.
#   pip install . pyinstaller
#   powershell -File packaging/build_exe.ps1

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
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
Write-Host "Done: dist/ecocorridor/ecocorridor.exe"
