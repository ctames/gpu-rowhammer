ptxas  -arch=sm_20 -m64  "sample.ptx"  -o "sample.sm_20.cubin"
fatbinary --create="sample.fatbin" -64 --key="xxxxxxxxxx" "--image=profile=sm_20,file=sample.sm_20.cubin" "--image=profile=compute_20,file=sample.ptx" --embedded-fatbin="sample.fatbin.c" --cuda
gcc -D__CUDA_ARCH__=200 -E -x c++        -DCUDA_DOUBLE_MATH_FUNCTIONS   -D__CUDA_PREC_DIV -D__CUDA_PREC_SQRT "-I/usr/local/cuda/bin/..//include"   -m64 "sample.cudafe1.cpp" > "sample.cu.cpp.ii"
gcc -c -x c++ "-I/usr/local/cuda/bin/..//include"   -fpreprocessed -m64 -o "sample.o" "sample.cu.cpp.ii"
nvlink --arch=sm_20 --register-link-binaries="sample_dlink.reg.c" -m64   "-L/usr/local/cuda/bin/..//lib64/stubs" "-L/usr/local/cuda/bin/..//lib64" -cpu-arch=X86_64 "sample.o"  -o "sample_dlink.sm_20.cubin"
fatbinary --create="sample_dlink.fatbin" -64 --key="sample_dlink" -link "--image=profile=sm_20,file=sample_dlink.sm_20.cubin" --embedded-fatbin="sample_dlink.fatbin.c"
gcc -c -x c++ -DFATBINFILE="\"sample_dlink.fatbin.c\"" -DREGISTERLINKBINARYFILE="\"sample_dlink.reg.c\"" -I. "-I/usr/local/cuda/bin/..//include"   -D"__CUDACC_VER__=70517" -D"__CUDACC_VER_BUILD__=17" -D"__CUDACC_VER_MINOR__=5" -D"__CUDACC_VER_MAJOR__=7" -m64 -o "sample_dlink.o" "/usr/local/cuda/bin/crt/link.stub"
g++ -m64 -o "sample" -Wl,--start-group "sample_dlink.o" "sample.o"   "-L/usr/local/cuda/bin/..//lib64/stubs" "-L/usr/local/cuda/bin/..//lib64" -lcudadevrt  -lcudart_static  -lrt -lpthread  -ldl  -Wl,--end-group
