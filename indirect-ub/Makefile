all: indirect

indirect: indirect.cu snappy.c 
	nvcc -arch sm_20 -I ~/.local/include/ -L ~/.local/lib/ $^ -o $@ -lsnappy

indirect_l: indirect.cu
	nvcc -arch sm_20 $< -o $@

