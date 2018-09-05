all: dgemm

dgemm: dgemm.cu
	nvcc -arch=sm_70 $< -o $@ -lcublas

clean:
	rm -rf dgemm

