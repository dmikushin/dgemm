#include <cstdio>
#include <cstdlib>
#include <cublas_v2.h>
#include <time.h>
#include <thrust/device_vector.h>
#include <vector>

using namespace std;

// Get the timer value.
static void get_time(double* ret)
{
	volatile struct timespec val;
	clock_gettime(CLOCK_REALTIME, (struct timespec*)&val);
	*ret = (double)0.000000001 * val.tv_nsec + val.tv_sec;
}

int main(int argc, char* argv[])
{
	int n = atoi(argv[1]);

	vector<double> A(n * n), B(n * n), C(n * n);

	const double dirandmax = 1.0 / RAND_MAX;
	for (int i = 0; i < n * n; i++)
	{
		A[i] = rand() * dirandmax;
		B[i] = rand() * dirandmax;
		C[i] = rand() * dirandmax;
	}

	const double alpha = rand() * dirandmax;
	const double beta = rand() * dirandmax;

	thrust::device_vector<double> dA = A;
	thrust::device_vector<double> dB = B;
	thrust::device_vector<double> dC = C;

	cublasHandle_t handle;
	cublasStatus_t cublasErr = cublasCreate(&handle);
	if (cublasErr != CUBLAS_STATUS_SUCCESS)
	{
		printf("Error creating CUBLAS context: err = %d!\n", cublasErr);
		exit(-1);
	}

	const int szbatch = 1000;
	while (1)
	{
		double start; get_time(&start);
		for (int i = 0; i < szbatch; i++)
		{
			cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, n, n, n,
				&alpha, thrust::raw_pointer_cast(&dA[0]), n,
				thrust::raw_pointer_cast(&dB[0]), n, &beta,
				thrust::raw_pointer_cast(&dC[0]), n);
			cudaError_t err = cudaDeviceSynchronize();
			if (err != cudaSuccess)
			{
				printf("CUDA error: err = %d\n", err);
				exit(-1);
			}
		}
		double finish; get_time(&finish);

		const double time = finish - start;
		printf("%f GFLOPS\n", (double)n * n * (2 * n + 3) / (1000 * 1000 * 1000 * time) * szbatch);
	}

	cublasDestroy(handle);

	return 0;
}
