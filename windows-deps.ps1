$ErrorActionPreference = "Stop"

# choco install miniforge3 --params="'/AddToPath:1 /InstallationType=JustMe /RegisterPython=1 /S /D=%UserProfile%\Miniforge3'"
# choco install llvm
# choco install cmake

& "$ENV:CONDA\shell\condabin\conda-hook.ps1"
conda env create -f psnr_hvsm-dev.yml


# Invoke-WebRequest -Uri https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Windows-x86_64.exe -OutFile Miniforge3-Windows-x86_64.exe
# Start-Process -FilePAth Miniforge3-Windows-x86_64.exe -ArgumentList "/InstallationType=JustMe /RegisterPython=1 /S /D=%UserProfile%\Miniforge3" -Wait
# & "$ENV:UserProfile\Miniforge3\shell\condabin\conda-hook.ps1"
# conda env create -f psnr_hvsm-dev.yml -q --force

# Invoke-WebRequest -Uri https://github.com/Kitware/CMake/releases/download/v3.27.5/cmake-3.27.5-windows-x86_64.msi -OutFile cmake.msi
# Start-Process msiexec.exe -ArgumentList "/i cmake.msi ADD_CMAKE_TO_PATH=User /qn" -Wait
