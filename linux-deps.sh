#!/bin/bash
set -ex

curl -L https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh --output Miniforge3.sh
bash Miniforge3.sh -b -p "${HOME}/.conda"
source "${HOME}/.conda/etc/profile.d/conda.sh"
conda activate
conda env create -f psnr_hvsm-dev.yml -q -y
conda activate psnr_hvsm-dev

curl -L https://github.com/Kitware/CMake/releases/download/v3.27.5/cmake-3.27.5-linux-x86_64.sh --output cmake.sh
bash cmake.sh --skip-license --prefix=/usr