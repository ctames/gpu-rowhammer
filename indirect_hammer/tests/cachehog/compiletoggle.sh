./cleanup.sh
rm *ptx 
nvcc -O0 -Xcicc -O0 -Xptxas -O0 -keep -o toggle toggle.cu
