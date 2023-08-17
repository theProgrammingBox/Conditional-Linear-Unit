#include "CLU.cuh"

int main()
{
	cublasStatus_t cublasStatus;
	cublasHandle_t cublasHandle;
	cublasStatus = cublasCreate(&cublasHandle);
	if (cublasStatus != CUBLAS_STATUS_SUCCESS) {
		printf("cublasCreate failed with error code %d\n", cublasStatus);
		return EXIT_FAILURE;
	}

	curandStatus_t curandStatus;
	curandGenerator_t curandGenerator;
	curandStatus = curandCreateGenerator(&curandGenerator, CURAND_RNG_PSEUDO_DEFAULT);
	if (curandStatus != CURAND_STATUS_SUCCESS) {
		printf("curandCreateGenerator failed with error code %d\n", curandStatus);
		return EXIT_FAILURE;
	}

	curandStatus = curandSetPseudoRandomGeneratorSeed(curandGenerator, 1234ULL);
	if (curandStatus != CURAND_STATUS_SUCCESS) {
		printf("curandSetPseudoRandomGeneratorSeed failed with error code %d\n", curandStatus);
		return EXIT_FAILURE;
	}

	float learningrate = 0.01f;
	size_t batches = 16;

	CLU clu
	(
		&cublasHandle, &curandGenerator, &learningrate,
		&batches, 16, 1, 8, 1
	);

	printf("CLU\n");

	return 0;
}