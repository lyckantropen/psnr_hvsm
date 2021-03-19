# dependency resolution
# requires a compiler, CMake and Git
# virtualenv is used to satisfy ancient NumPy detection for xtensor-python

param(
    [string]$Destination = ".third_party",
    [string]$PythonPath = $null
)

$ErrorActionPreference = "Stop"

if($PythonPath -eq [String]::Empty) {
    $PythonPath = If($IsWindows) {"py -3"} else {"/opt/python/cp37-cp37m/bin/python"}
}
Write-Output "Using Python: $PythonPath"
Invoke-Expression "$PythonPath -m pip install virtualenv"

function Build-And-Install-CMake-Package($SourceFolder, $Dest, $CMakeConfigureOptions="", $Rebuild=$true) {
    Push-Location $SourceFolder -ErrorAction Stop
        if ((Test-Path build) -And $Rebuild -eq $true) {
            Write-Output "Removing $SourceFolder/build"
            Remove-Item -r -fo build
        }
        New-Item build -ItemType Directory -ea 0
        Push-Location build
            $arch = If ($IsWindows) {"-Ax64"} else {""}
            Write-Output "cmake .. $CMakeConfigureOptions $arch -DCMAKE_INSTALL_PREFIX=$Dest"
            cmake .. "$CMakeConfigureOptions $arch".split(' ') -DCMAKE_INSTALL_PREFIX="$Dest"
            cmake --build . --config Release -j4
            cmake --install .
        Pop-Location
    Pop-Location
}

function Install-Package-From-Git($Url, $Ref, $SourceFolder, $Dest, $CMakeConfigureOptions="") {
    if (-Not (Test-Path $SourceFolder)) {
        git clone $Url $SourceFolder
    }

    Push-Location $SourceFolder -ErrorAction Stop
        git checkout $Ref
    Pop-Location
    Build-And-Install-CMake-Package -SourceFolder $SourceFolder -Dest $Dest -CMakeConfigureOptions $CMakeConfigureOptions
}

New-Item $Destination -ItemType Directory -ea 0
$ThirdPartyFolder = (Resolve-Path $Destination)

Write-Output "ThirdPartyFolder=$ThirdPartyFolder"

New-Item $ThirdPartyFolder/src -ItemType Directory -ea 0

Push-Location $ThirdPartyFolder/src
    # obtain and build FFTW3 (static libraries)
    if(-Not (Test-Path "fftw-3.3.9.tar.gz")) {
        Invoke-WebRequest -Uri http://www.fftw.org/fftw-3.3.9.tar.gz -OutFile fftw-3.3.9.tar.gz
    }
    if(-Not (Test-Path "fftw-3.3.9")) {
        tar -zxf fftw-3.3.9.tar.gz
    }
    Build-And-Install-CMake-Package -SourceFolder fftw-3.3.9 -Dest $ThirdPartyFolder -CMakeConfigureOptions '-DBUILD_SHARED_LIBS=OFF -DBUILD_TESTS=OFF -DCMAKE_POSITION_INDEPENDENT_CODE=ON' -Rebuild $false
    Build-And-Install-CMake-Package -SourceFolder fftw-3.3.9 -Dest $ThirdPartyFolder -CMakeConfigureOptions '-DBUILD_SHARED_LIBS=OFF -DBUILD_TESTS=OFF -DENABLE_FLOAT=ON -DCMAKE_POSITION_INDEPENDENT_CODE=ON' -Rebuild $false
    If($IsLinux) {
        Build-And-Install-CMake-Package -SourceFolder fftw-3.3.9 -Dest $ThirdPartyFolder -CMakeConfigureOptions '-DBUILD_SHARED_LIBS=OFF -DBUILD_TESTS=OFF -DENABLE_LONG_DOUBLE=ON -DCMAKE_POSITION_INDEPENDENT_CODE=ON' -Rebuild $false
    }

    # xtensor-stack libraries
    Install-Package-From-Git -Url https://github.com/xtensor-stack/xtl.git -Ref tags/0.7.2 -SourceFolder xtl -Dest $ThirdPartyFolder
    Install-Package-From-Git -Url https://github.com/xtensor-stack/xtensor.git -Ref tags/0.23.2 -SourceFolder xtensor -Dest $ThirdPartyFolder
    Install-Package-From-Git -Url https://github.com/xtensor-stack/xsimd.git -Ref tags/7.4.9 -SourceFolder xsimd -Dest $ThirdPartyFolder
    Install-Package-From-Git -Url https://github.com/xtensor-stack/xtensor-fftw.git -Ref tags/0.2.6 -SourceFolder xtensor-fftw -Dest $ThirdPartyFolder

    # we need to activate a virtual environment for xtensor-python with numpy in it
    # the version of numpy doesn't matter
    Invoke-Expression "$PythonPath -m virtualenv .venv"
    If($IsWindows) {
        ./.venv/Scripts/activate.ps1    
    }
    else {
    ./.venv/bin/activate.ps1
    }
        # pybind11
        Install-Package-From-Git -Url https://github.com/pybind/pybind11.git -Ref tags/v2.6.2 -SourceFolder pybind11 -Dest $ThirdPartyFolder -CMakeConfigureOptions -DPYBIND11_TEST=OFF
        # xtensor-python
        pip install numpy
        Install-Package-From-Git -Url https://github.com/xtensor-stack/xtensor-python.git -Ref tags/0.25.1 -SourceFolder xtensor-python -Dest $ThirdPartyFolder
    deactivate
Pop-Location