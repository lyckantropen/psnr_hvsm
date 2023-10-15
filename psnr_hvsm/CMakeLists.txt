set(BUILD_SHARED_LIBS OFF CACHE INTERNAL "FFTW3: Shared libs off")
set(BUILD_TESTS OFF CACHE INTERNAL "FFTW3: Tests off")

include(FetchContent)
FetchContent_Declare(
  fftw3
  URL       http://fftw.org/fftw-3.3.10.tar.gz
  URL_HASH  MD5=8ccbf6a5ea78a16dbc3e1306e234cc5c
  DOWNLOAD_EXTRACT_TIMESTAMP TRUE
)
FetchContent_MakeAvailable(fftw3)

find_package(Python3 COMPONENTS Interpreter NumPy Development.Module REQUIRED)
find_package(pybind11 REQUIRED)
find_package(xtensor REQUIRED)
find_package(xtensor-python REQUIRED)
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