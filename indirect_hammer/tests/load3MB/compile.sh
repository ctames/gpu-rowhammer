./cleanup.sh
rm *ptx 
nvcc -O0 -Xcicc -O0 -Xptxas -O0 -keep -o load3MB load3MB.cu
