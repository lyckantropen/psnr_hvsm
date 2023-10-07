$ErrorActionPreference = "Stop"

& "$ENV:CONDA\shell\condabin\conda-hook.ps1"
conda env create -f psnr_hvsm-dev.yml
