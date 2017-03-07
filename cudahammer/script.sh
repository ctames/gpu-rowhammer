#$ _SPACE_= 
#$ _CUDART_=cudart
#$ _HERE_=/usr/local/cuda/bin
#$ _THERE_=/usr/local/cuda/bin
#$ _TARGET_SIZE_=
#$ _TARGET_DIR_=
#$ _TARGET_DIR_=targets/x86_64-linux
#$ TOP=/usr/local/cuda/bin/..
#$ NVVMIR_LIBRARY_DIR=/usr/local/cuda/bin/../nvvm/libdevice
#$ LD_LIBRARY_PATH=/usr/local/cuda/bin/../lib:
#$ PATH=/usr/local/cuda/bin/../open64/bin:/usr/local/cuda/bin/../nvvm/bin:/usr/local/cuda/bin:/usr/lib64/qt-3.3/bin:/home/corey/opt/Qt5.1.1/5.1.1/gcc_64/bin:/usr/local/bin:/usr/bin:/usr/local/sbin:/usr/sbin:/usr/local/torque-5.1.2/bin:/usr/local/cuda/bin:/home/corey/.local:/home/corey/.local/bin:/home/corey/bin
#$ INCLUDES="-I/usr/local/cuda/bin/../targets/x86_64-linux/include"  
#$ LIBRARIES=  "-L/usr/local/cuda/bin/../targets/x86_64-linux/lib/stubs" "-L/usr/local/cuda/bin/../targets/x86_64-linux/lib"
#$ CUDAFE_FLAGS=
#$ PTXAS_FLAGS=
#$ rm cudahammer_dlink.reg.c
#$ gcc -D__CUDA_ARCH__=200 -E -x c++        -DCUDA_DOUBLE_MATH_FUNCTIONS  -D__CUDACC__ -D__NVCC__  "-I/usr/local/cuda/bin/../targets/x86_64-linux/include"   -D"__CUDACC_VER__=70517" -D"__CUDACC_VER_BUILD__=17" -D"__CUDACC_VER_MINOR__=5" -D"__CUDACC_VER_MAJOR__=7" -include "cuda_runtime.h" -m64 "cudahammer_test.cu" > "cudahammer_test.cpp1.ii" 
#$ cudafe --allow_managed --m64 --gnu_version=40805 -tused --no_remove_unneeded_entities  --gen_c_file_name "cudahammer_test.cudafe1.c" --stub_file_name "cudahammer_test.cudafe1.stub.c" --gen_device_file_name "cudahammer_test.cudafe1.gpu" --nv_arch "compute_20" --gen_module_id_file --module_id_file_name "cudahammer_test.module_id" --include_file_name "cudahammer_test.fatbin.c" "cudahammer_test.cpp1.ii" 
#$ gcc -D__CUDA_ARCH__=200 -E -x c        -DCUDA_DOUBLE_MATH_FUNCTIONS  -D__CUDACC__ -D__NVCC__ -D__CUDANVVM__  -D__CUDA_PREC_DIV -D__CUDA_PREC_SQRT "-I/usr/local/cuda/bin/../targets/x86_64-linux/include"   -m64 "cudahammer_test.cudafe1.gpu" > "cudahammer_test.cpp2.i" 
#$ cudafe -w --allow_managed --m64 --gnu_version=40805 --c  --gen_c_file_name "cudahammer_test.cudafe2.c" --stub_file_name "cudahammer_test.cudafe2.stub.c" --gen_device_file_name "cudahammer_test.cudafe2.gpu" --nv_arch "compute_20" --module_id_file_name "cudahammer_test.module_id" --include_file_name "cudahammer_test.fatbin.c" "cudahammer_test.cpp2.i" 
#$ gcc -D__CUDA_ARCH__=200 -E -x c        -DCUDA_DOUBLE_MATH_FUNCTIONS  -D__CUDABE__ -D__CUDANVVM__  -D__CUDA_PREC_DIV -D__CUDA_PREC_SQRT "-I/usr/local/cuda/bin/../targets/x86_64-linux/include"   -m64 "cudahammer_test.cudafe2.gpu" > "cudahammer_test.cpp3.i" 
#$ filehash -s " " "cudahammer_test.cpp3.i" > "cudahammer_test.hash"
#$ gcc -E -x c++ -D__CUDACC__ -D__NVCC__  "-I/usr/local/cuda/bin/../targets/x86_64-linux/include"   -D"__CUDACC_VER__=70517" -D"__CUDACC_VER_BUILD__=17" -D"__CUDACC_VER_MINOR__=5" -D"__CUDACC_VER_MAJOR__=7" -include "cuda_runtime.h" -m64 "cudahammer_test.cu" > "cudahammer_test.cpp4.ii" 
#$ cudafe++ --allow_managed --m64 --gnu_version=40805 --parse_templates  --gen_c_file_name "cudahammer_test.cudafe1.cpp" --stub_file_name "cudahammer_test.cudafe1.stub.c" --module_id_file_name "cudahammer_test.module_id" "cudahammer_test.cpp4.ii" 
#$ cicc  -arch compute_20 -m64 -ftz=0 -prec_div=1 -prec_sqrt=1 -fmad=1 -nvvmir-library "/usr/local/cuda/bin/../nvvm/libdevice/libdevice.compute_20.10.bc" --orig_src_file_name "cudahammer_test.cu"  "cudahammer_test.cpp3.i" -o "cudahammer_test.ptx"
#$ ptxas  -arch=sm_20 -m64  "cudahammer_test.ptx"  -o "cudahammer_test.sm_20.cubin" 
fatbinary --create="cudahammer_test.fatbin" -64 --key="xxxxxxxxxx" "--image=profile=sm_20,file=cudahammer_test.sm_20.cubin" "--image=profile=compute_20,file=cudahammer_test.ptx" --embedded-fatbin="cudahammer_test.fatbin.c" --cuda
gcc -D__CUDA_ARCH__=200 -E -x c++        -DCUDA_DOUBLE_MATH_FUNCTIONS   -D__CUDA_PREC_DIV -D__CUDA_PREC_SQRT "-I/usr/local/cuda/bin/../targets/x86_64-linux/include"   -m64 "cudahammer_test.cudafe1.cpp" > "cudahammer_test.cu.cpp.ii" 
gcc -c -x c++ "-I/usr/local/cuda/bin/../targets/x86_64-linux/include"   -fpreprocessed -m64 -o "cudahammer_test.o" "cudahammer_test.cu.cpp.ii" 
nvlink --arch=sm_20 --register-link-binaries="cudahammer_dlink.reg.c" -m64   "-L/usr/local/cuda/bin/../targets/x86_64-linux/lib/stubs" "-L/usr/local/cuda/bin/../targets/x86_64-linux/lib" -cpu-arch=X86_64 "cudahammer_test.o"  -o "cudahammer_dlink.sm_20.cubin"
fatbinary --create="cudahammer_dlink.fatbin" -64 --key="cudahammer_dlink" -link "--image=profile=sm_20,file=cudahammer_dlink.sm_20.cubin" --embedded-fatbin="cudahammer_dlink.fatbin.c" 
gcc -c -x c++ -DFATBINFILE="\"cudahammer_dlink.fatbin.c\"" -DREGISTERLINKBINARYFILE="\"cudahammer_dlink.reg.c\"" -I. "-I/usr/local/cuda/bin/../targets/x86_64-linux/include"   -D"__CUDACC_VER__=70517" -D"__CUDACC_VER_BUILD__=17" -D"__CUDACC_VER_MINOR__=5" -D"__CUDACC_VER_MAJOR__=7" -m64 -o "cudahammer_dlink.o" "/usr/local/cuda/bin/crt/link.stub" 
g++ -m64 -o "cudahammer" -Wl,--start-group "cudahammer_dlink.o" "cudahammer_test.o"   "-L/usr/local/cuda/bin/../targets/x86_64-linux/lib/stubs" "-L/usr/local/cuda/bin/../targets/x86_64-linux/lib" -lcudadevrt  -lcudart_static  -lrt -lpthread  -ldl  -Wl,--end-group 
