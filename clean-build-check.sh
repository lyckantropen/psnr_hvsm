#!/usr/bin/env bash
#
# Build this package from source in a clean environment and run the test suite
# against the result.
#
# A "clean environment" here means: no psnr_hvsm-dev conda/Miniforge environment
# anywhere CMake looks for it, no third-party CMake packages installed for CMake
# to discover, and a fresh virtualenv that only knows the declared build
# requirements. The header-only C++ dependencies therefore have to be fetched,
# which is the path a contributor without a conda installation gets.
#
# Three things are checked:
#   1. CMake 4.x configures the project with no policy errors (the fetched FFTW
#      release predates CMake 4's compatibility removal) and reports that it is
#      fetching the header-only dependencies.
#   2. pip installs the package from source in that clean environment.
#   3. The test suite passes against the installed result.
#
# Environment variables:
#   PYTHON       interpreter used to create the virtualenv (default: python3)
#   WITH_TORCH   install PyTorch before running the suite (default: 1);
#                set to 0 where PyTorch is unavailable; torch tests are
#                collected and skipped by their optional-dependency guards
#   REQUIRE_CMAKE4  fail unless the CMake on PATH is at least 4.0 (default: 1)
#   KEEP         keep the scratch directory for inspection (default: 0)

set -eo pipefail

REPO="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="${PYTHON:-python3}"
WITH_TORCH="${WITH_TORCH:-1}"
REQUIRE_CMAKE4="${REQUIRE_CMAKE4:-1}"
KEEP="${KEEP:-0}"
TORCH_INDEX="https://download.pytorch.org/whl/cpu"

step() { printf '\n=== %s ===\n' "$*"; }
fail() { printf 'clean-build-check: %s\n' "$*" >&2; exit 1; }

command -v cmake >/dev/null 2>&1 || fail "cmake not found on PATH"
command -v "$PYTHON" >/dev/null 2>&1 || fail "python interpreter '$PYTHON' not found on PATH"

cmake_version="$(cmake --version | head -1 | grep -o '[0-9]*\.[0-9]*\.[0-9]*')"
step "CMake $cmake_version"
if [ "$REQUIRE_CMAKE4" = 1 ] && [ "$(printf '%s' "$cmake_version" | cut -d. -f1)" -lt 4 ]; then
  fail "this check builds under CMake 4.x, found $cmake_version (set REQUIRE_CMAKE4=0 to skip)"
fi

WORK="${TMPDIR:-/tmp}/psnr_hvsm-clean-build-check"
rm -rf "$WORK"
mkdir -p "$WORK"

# The conda discovery in CMakeLists.txt looks for these exact directories.
for env_dir in \
    "${HOME}/.conda/envs/psnr_hvsm-dev" \
    "${HOME}/Miniforge3/envs/psnr_hvsm-dev" \
    "${CONDA:-$WORK/no-conda}/envs/psnr_hvsm-dev"; do
  [ -d "$env_dir" ] && fail "conda development environment found at '$env_dir', this check needs a clean environment"
done

# Point HOME somewhere empty so the conda discovery in CMakeLists.txt cannot
# find anything even when the machine does have a conda installation.
export HOME="$WORK/fake-home"
mkdir -p "$HOME"
unset CONDA

step "creating a virtualenv"
"$PYTHON" -m venv "$WORK/venv"
# shellcheck disable=SC2016
"$WORK/venv/bin/python" -m pip install --disable-pip-version-check --quiet --upgrade pip

# CMake needs the NumPy headers of this interpreter to configure the extension.
"$WORK/venv/bin/python" -m pip install --disable-pip-version-check --quiet numpy

step "configuring the project with the CMake on PATH"
# A plain CMake configure has to succeed outside of pip's build isolation. The
# interpreter is pointed at the virtualenv so that CMake finds NumPy there, and
# NumPy's include directory is passed explicitly because the vendored xtensor
# CMake scripts look for it through the legacy PythonInterp variables.
# Discovery of the header-only dependencies is switched off so that CMake is
# forced down the fetch path this check is about, even on a machine where the
# distribution or the system interpreter happen to ship those packages.
# Search prefixes for the interpreter are ignored wholesale, because a pip
# pybind11 installed into a site-packages directory is discovered through them
# and its exported include paths are wrong for a virtualenv interpreter.
numpy_include="$("$WORK/venv/bin/python" -c 'import numpy; print(numpy.get_include())')"
cmake \
  -S "$REPO" \
  -B "$WORK/build" \
  -DPython3_EXECUTABLE="$WORK/venv/bin/python" \
  -DNUMPY_INCLUDE_DIRS="$numpy_include" \
  -DCMAKE_DISABLE_FIND_PACKAGE_pybind11=TRUE \
  -DCMAKE_DISABLE_FIND_PACKAGE_xtensor=TRUE \
  -DCMAKE_DISABLE_FIND_PACKAGE_xtensor-python=TRUE \
  -DCMAKE_BUILD_TYPE=RelWithDebInfo \
  -DCMAKE_POSITION_INDEPENDENT_CODE=ON \
  >"$WORK/configure.log" 2>&1 || {
    tail -40 "$WORK/configure.log"
    fail "CMake configure failed under CMake $cmake_version"
  }
tail -30 "$WORK/configure.log"

if grep -q "Compatibility with CMake < 3.5" "$WORK/configure.log"; then
  fail "the fetched FFTW still refuses to configure under CMake 4"
fi
if grep -q "CMake Error" "$WORK/configure.log"; then
  fail "CMake reported an error while configuring"
fi

# Nothing was installed for CMake to discover, so every header-only dependency
# has to be fetched. Assert that this really happened instead of trusting that
# the build would have failed otherwise.
grep -q "psnr_hvsm: pybind11 not found on the system, fetching" "$WORK/configure.log" \
  || fail "the configure log does not show pybind11 being fetched"
grep -q "psnr_hvsm: xtensor, xtensor-python not found on the system, fetching" "$WORK/configure.log" \
  || fail "the configure log does not show the xtensor stack being fetched"
echo "the header-only dependencies were fetched, not discovered"

step "installing the package from source with pip"
# The same two overrides as the configure above, repeated once per setting,
# passed into pip's build isolation through scikit-build-core's config settings.
config_settings=(
  --config-settings cmake.define.CMAKE_DISABLE_FIND_PACKAGE_pybind11=TRUE
  --config-settings cmake.define.CMAKE_DISABLE_FIND_PACKAGE_xtensor=TRUE
  --config-settings cmake.define.CMAKE_DISABLE_FIND_PACKAGE_xtensor-python=TRUE
)
"$WORK/venv/bin/python" -m pip install --disable-pip-version-check --quiet \
  "${config_settings[@]}" \
  "$REPO[tests]" >"$WORK/pip.log" 2>&1 || {
    tail -40 "$WORK/pip.log"
    fail "pip install from source failed"
  }

if [ "$WITH_TORCH" = 1 ]; then
  echo "installing PyTorch from $TORCH_INDEX"
  "$WORK/venv/bin/python" -m pip install --disable-pip-version-check --quiet \
    torch --index-url "$TORCH_INDEX" --extra-index-url https://pypi.org/simple \
    >>"$WORK/pip.log" 2>&1 || {
      tail -40 "$WORK/pip.log"
      fail "installing PyTorch failed (set WITH_TORCH=0 to skip it)"
    }
  "$WORK/venv/bin/python" -m pip install --disable-pip-version-check --quiet \
    torch-dct >>"$WORK/pip.log" 2>&1 \
    || fail "installing torch-dct failed"
else
  echo "WITH_TORCH=0, skipping PyTorch"
fi

step "the installed package"
# Run from the scratch directory, not the repository, so that every import
# resolves against the installed package rather than the source tree.
cd "$WORK"
"$WORK/venv/bin/python" - <<'PY'
import numpy
import scipy

import psnr_hvsm

print(f"psnr_hvsm from {psnr_hvsm.__file__}")
print(f"numpy {numpy.__version__}, scipy {scipy.__version__}")
print(f"backends: {sorted(psnr_hvsm.BACKENDS)}")
assert "cpp" in psnr_hvsm.BACKENDS, "the C++ backend did not build or did not import"
PY

step "running the test suite"
"$WORK/venv/bin/python" -m pytest "$REPO/tests" || fail "the test suite failed"

step "checking that installed dependencies are still discovered"
# The conda development environment supplies pybind11, xtl, xsimd, xtensor and
# xtensor-python as installed CMake packages. Simulate one by installing the
# pinned dependencies fetched above into a prefix directory that the top-level
# CMakeLists looks for under a fake HOME, and configure again: CMake must
# discover them instead of fetching anything.
CONDA_PREFIX="${WORK}/fake-home/.conda/envs/psnr_hvsm-dev"
cmake \
  -S "${WORK}/build/_deps/pybind11-src" \
  -B "${WORK}/build-pybind11-install" \
  -DPYBIND11_INSTALL=ON \
  -DPYBIND11_TEST=OFF \
  -DPython_EXECUTABLE="$WORK/venv/bin/python" \
  >"$WORK/configure-pybind11-install.log" 2>&1 || {
    tail -40 "$WORK/configure-pybind11-install.log"
    fail "configuring the fetched pybind11 for installation failed"
  }
cmake --install "${WORK}/build-pybind11-install" --prefix "$CONDA_PREFIX" >/dev/null

for dependency in xtl xsimd xtensor xtensor-python; do
  cmake --install "${WORK}/build/_deps/${dependency}-build" --prefix "$CONDA_PREFIX" >/dev/null
done
HOME="$WORK/fake-home" CONDA= cmake \
  -S "$REPO" \
  -B "$WORK/build-discovered" \
  -DPython3_EXECUTABLE="$WORK/venv/bin/python" \
  -DNUMPY_INCLUDE_DIRS="$numpy_include" \
  -DFETCHCONTENT_SOURCE_DIR_FFTW3="${WORK}/build/_deps/fftw3-src" \
  >"$WORK/configure-discovered.log" 2>&1 || {
    tail -40 "$WORK/configure-discovered.log"
    fail "CMake configure with installed dependencies failed"
  }
grep -E "psnr_hvsm:" "$WORK/configure-discovered.log"

if grep -q "not found on the system, fetching" "$WORK/configure-discovered.log"; then
  fail "a header-only dependency was fetched although an installed one was available"
fi
HOME="$WORK/fake-home" CONDA= make -C "$WORK/build-discovered" -j8 \
  -f psnr_hvsm/CMakeFiles/_psnr_hvsm.dir/build.make \
  psnr_hvsm/CMakeFiles/_psnr_hvsm.dir/dct.cpp.o \
  >"$WORK/build-discovered.log" 2>&1 || {
    tail -30 "$WORK/build-discovered.log"
    fail "compiling against the discovered dependencies failed"
  }
echo "discovered the installed dependencies instead of fetching them"

step "ok"
echo "logs kept in $WORK (set KEEP=1 to inspect them)"
if [ "$KEEP" != 1 ]; then
  rm -rf "$WORK"
fi
