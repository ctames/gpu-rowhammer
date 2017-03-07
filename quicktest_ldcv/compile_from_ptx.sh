ptxas  -arch=sm_20 -m64  "test.ptx"  -o "test.sm_20.cubin" 
fatbinary --create="test.fatbin" -64 --key="xxxxxxxxxx" "--image=profile=sm_20,file=test.sm_20.cubin" "--image=profile=compute_20,file=test.ptx" --embedded-fatbin="test.fatbin.c" --cuda
gcc -D__CUDA_ARCH__=200 -E -x c++        -DCUDA_DOUBLE_MATH_FUNCTIONS   -D__CUDA_PREC_DIV -D__CUDA_PREC_SQRT "-I/usr/local/cuda/bin/../targets/x86_64-linux/include"   -m64 "test.cudafe1.cpp" > "test.cu.cpp.ii" 
gcc -c -x c++ "-I/usr/local/cuda/bin/../targets/x86_64-linux/include"   -fpreprocessed -m64 -o "test.o" "test.cu.cpp.ii" 
nvlink --arch=sm_20 --register-link-binaries="test_dlink.reg.c" -m64   "-L/usr/local/cuda/bin/../targets/x86_64-linux/lib/stubs" "-L/usr/local/cuda/bin/../targets/x86_64-linux/lib" -cpu-arch=X86_64 "test.o"  -o "test_dlink.sm_20.cubin"
fatbinary --create="test_dlink.fatbin" -64 --key="test_dlink" -link "--image=profile=sm_20,file=test_dlink.sm_20.cubin" --embedded-fatbin="test_dlink.fatbin.c" 
gcc -c -x c++ -DFATBINFILE="\"test_dlink.fatbin.c\"" -DREGISTERLINKBINARYFILE="\"test_dlink.reg.c\"" -I. "-I/usr/local/cuda/bin/../targets/x86_64-linux/include"   -D"__CUDACC_VER__=70517" -D"__CUDACC_VER_BUILD__=17" -D"__CUDACC_VER_MINOR__=5" -D"__CUDACC_VER_MAJOR__=7" -m64 -o "test_dlink.o" "/usr/local/cuda/bin/crt/link.stub" 
g++ -m64 -o "test" -Wl,--start-group "test_dlink.o" "test.o"   "-L/usr/local/cuda/bin/../targets/x86_64-linux/lib/stubs" "-L/usr/local/cuda/bin/../targets/x86_64-linux/lib" -lcudadevrt  -lcudart_static  -lrt -lpthread  -ldl  -Wl,--end-group 
