find_package(Python3 COMPONENTS Interpreter NumPy Development.Module REQUIRED)
find_package(pybind11 REQUIRED)
find_package(xtensor REQUIRED)
find_package(xtensor-python REQUIRED)
find_package(FFTW3 COMPONENTS DOUBLE_LIB REQUIRED)
find_package(OpenMP COMPONENTS CXX)

set(PSNR_HVSM_FILES
    dct.cpp
    psnr_hvsm.cpp
    psnr_hvsm_pybind11.cpp
)

pybind11_add_module(_psnr_hvsm ${PSNR_HVSM_FILES})
target_compile_features(_psnr_hvsm PRIVATE cxx_std_17)
target_compile_definitions(_psnr_hvsm PRIVATE -D_USE_MATH_DEFINES)
target_include_directories(_psnr_hvsm PRIVATE ${xtensor_INCLUDE_DIRS} ${xtensor-python_INCLUDE_DIRS})
target_link_libraries(_psnr_hvsm PRIVATE
                                 FFTW3::fftw3
                                 pybind11::module
                                 Python3::Module
                                 Python3::NumPy
                                 xtensor::use_xsimd
)

if(OpenMP_FOUND)
    target_compile_definitions(_psnr_hvsm PRIVATE -DPSNR_HVSM_OPENMP -DXTENSOR_USE_OPENMP)
    target_link_libraries(_psnr_hvsm PRIVATE OpenMP::OpenMP_CXX)
endif()

install(TARGETS _psnr_hvsm LIBRARY DESTINATION psnr_hvsm)