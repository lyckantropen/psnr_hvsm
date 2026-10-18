set(BUILD_SHARED_LIBS OFF CACHE INTERNAL "FFTW3: Shared libs off")
set(BUILD_TESTS OFF CACHE INTERNAL "FFTW3: Tests off")

# FFTW 3.3.10 declares cmake_minimum_required(VERSION 3.0), which CMake 4
# removed compatibility with outright. Raising the policy version for the
# third-party projects added below lets the fetched FFTW configure under CMake 4
# without patching its sources and without downgrading CMake. CMake versions
# that predate the variable do not know it and ignore it, so an older CMake
# keeps configuring the project.
set(CMAKE_POLICY_VERSION_MINIMUM 3.5)

include(FetchContent)

FetchContent_Declare(
  fftw3
  URL       http://fftw.org/fftw-3.3.10.tar.gz
  URL_HASH  MD5=8ccbf6a5ea78a16dbc3e1306e234cc5c
  DOWNLOAD_EXTRACT_TIMESTAMP TRUE
)
FetchContent_MakeAvailable(fftw3)

# Pinned versions of the header-only C++ dependency stack, fetched when the
# dependencies are not installed. The set is mutually compatible:
# xtensor-python 0.27.0 requires xtensor >= 0.25.0 and pybind11 >= 2.6.1, and
# xtensor 0.25.0 requires xtl >= 0.7.5 and xsimd >= 11.0.0. All of them keep
# the flat xtensor/xtensor.hpp umbrella header this project's sources include.
set(PSNR_HVSM_PYBIND11_VERSION v2.13.6)
set(PSNR_HVSM_XTL_VERSION 0.7.7)
set(PSNR_HVSM_XSIMD_VERSION 13.2.0)
set(PSNR_HVSM_XTENSOR_VERSION 0.25.0)
set(PSNR_HVSM_XTENSOR_PYTHON_VERSION 0.27.0)

find_package(Python3 COMPONENTS Interpreter NumPy Development.Module REQUIRED)
find_package(pybind11 QUIET)
find_package(xtensor QUIET)
find_package(xtensor-python QUIET)

if(pybind11_FOUND)
  message(STATUS "psnr_hvsm: using pybind11 ${pybind11_VERSION} found on the system")
else()
  message(STATUS "psnr_hvsm: pybind11 not found on the system, fetching ${PSNR_HVSM_PYBIND11_VERSION}")
  FetchContent_Declare(
    pybind11
    GIT_REPOSITORY https://github.com/pybind/pybind11.git
    GIT_TAG        ${PSNR_HVSM_PYBIND11_VERSION}
    GIT_SHALLOW    TRUE
    # EXCLUDE_FROM_ALL is required for two reasons: the fetched projects are
    # header-only, so nothing of theirs needs building, and it stops their
    # install rules from copying headers into the wheel.
    EXCLUDE_FROM_ALL
  )
endif()

if(xtensor_FOUND AND xtensor-python_FOUND)
  message(STATUS "psnr_hvsm: using xtensor ${xtensor_VERSION} and xtensor-python ${xtensor-python_VERSION} found on the system")
else()
  message(STATUS "psnr_hvsm: xtensor, xtensor-python not found on the system, fetching xtl ${PSNR_HVSM_XTL_VERSION}, xsimd ${PSNR_HVSM_XSIMD_VERSION}, xtensor ${PSNR_HVSM_XTENSOR_VERSION}, xtensor-python ${PSNR_HVSM_XTENSOR_PYTHON_VERSION}")
  FetchContent_Declare(
    xtl
    GIT_REPOSITORY https://github.com/xtensor-stack/xtl.git
    GIT_TAG        ${PSNR_HVSM_XTL_VERSION}
    GIT_SHALLOW    TRUE
    EXCLUDE_FROM_ALL
  )
  FetchContent_Declare(
    xsimd
    GIT_REPOSITORY https://github.com/xtensor-stack/xsimd.git
    GIT_TAG        ${PSNR_HVSM_XSIMD_VERSION}
    GIT_SHALLOW    TRUE
    EXCLUDE_FROM_ALL
  )
  FetchContent_Declare(
    xtensor
    GIT_REPOSITORY https://github.com/xtensor-stack/xtensor.git
    GIT_TAG        ${PSNR_HVSM_XTENSOR_VERSION}
    GIT_SHALLOW    TRUE
    EXCLUDE_FROM_ALL
  )
  FetchContent_Declare(
    xtensor-python
    GIT_REPOSITORY https://github.com/xtensor-stack/xtensor-python.git
    GIT_TAG        ${PSNR_HVSM_XTENSOR_PYTHON_VERSION}
    GIT_SHALLOW    TRUE
    EXCLUDE_FROM_ALL
  )
endif()

# Made available in dependency order, because xtensor resolves itself against
# the xtl target and xtensor-python against the xtensor and pybind11 targets.
if(NOT pybind11_FOUND)
  FetchContent_MakeAvailable(pybind11)
endif()

if(NOT (xtensor_FOUND AND xtensor-python_FOUND))
  FetchContent_MakeAvailable(xtl)
  FetchContent_MakeAvailable(xsimd)
  FetchContent_MakeAvailable(xtensor)
  FetchContent_MakeAvailable(xtensor-python)

  # xtensor's installed package config also defines the imported targets
  # xtensor::use_xsimd and xtensor::optimize, which its build tree does not
  # create. Reproduce the one this module links against, as the config does.
  if(NOT TARGET xtensor::use_xsimd)
    add_library(xtensor::use_xsimd INTERFACE IMPORTED)
    target_link_libraries(xtensor::use_xsimd INTERFACE xsimd)
    target_compile_definitions(xtensor::use_xsimd INTERFACE XTENSOR_USE_XSIMD)
  endif()
endif()

find_package(OpenMP COMPONENTS CXX)

set(PSNR_HVSM_FILES
    dct.cpp
    psnr_hvsm.cpp
    psnr_hvsm_pybind11.cpp
)

pybind11_add_module(_psnr_hvsm ${PSNR_HVSM_FILES})
target_compile_features(_psnr_hvsm PRIVATE cxx_std_17)
target_compile_definitions(_psnr_hvsm PRIVATE -D_USE_MATH_DEFINES)
target_include_directories(_psnr_hvsm PRIVATE ${fftw3_SOURCE_DIR}/api)
target_link_libraries(_psnr_hvsm PRIVATE
                                fftw3
                                pybind11::module
                                Python3::Module
                                Python3::NumPy
                                xtensor
                                xtensor-python
                                xtensor::use_xsimd
)


if(OpenMP_FOUND AND NOT MSVC)
    target_compile_definitions(_psnr_hvsm PRIVATE -DPSNR_HVSM_OPENMP -DXTENSOR_USE_OPENMP)
    target_link_libraries(_psnr_hvsm PRIVATE OpenMP::OpenMP_CXX)
endif()

install(FILES $<LIST:FILTER,$<TARGET_RUNTIME_DLLS:_psnr_hvsm>,INCLUDE,.*fftw.*> DESTINATION ${SKBUILD_PROJECT_NAME})
install(TARGETS _psnr_hvsm LIBRARY DESTINATION ${SKBUILD_PROJECT_NAME})
