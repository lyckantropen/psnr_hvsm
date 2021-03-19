$Env:PSNR_HVSM_EXTRA_CMAKE_ARGS='-DCMAKE_PREFIX_PATH=C:/Users/Karol Trojanowski/psnr_hvsm/.third_party,-DCMAKE_TOOLCHAIN_FILE=C:/vcpkg/scripts/buildsystems/vcpkg.cmake,-DVCPKG_TARGET_TRIPLET=x64-windows-static'
python setup.py bdist_wheel

python .\setup.py bdist_wheel -DCMAKE_PREFIX_PATH='C:/Users/Karol Trojanowski/psnr_hvsm/.third_party'

tox --parallel auto -- --extra-cmake-args '-DCMAKE_PREFIX_PATH=C:/Users/Karol Trojanowski/psnr_hvsm/.third_party,-DCMAKE_TOOLCHAIN_FILE=C:/vcpkg/scripts/buildsystems/vcpkg.cmake,-DVCPKG_TARGET_TRIPLET=x64-windows-static'