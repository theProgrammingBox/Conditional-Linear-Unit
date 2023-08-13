#include "NeuralNetwork.cuh"

/*
TODO:
- work on gpu
-- add dynamic input and outputGrad copy option by passing nullptr
-- see if you can make cpuSaxpy and cpuBinaryForward like cpuSgemmStridedBatched
-- make inHeight allocation dynamic by calculating the max it can allocate and ensure it is not exceeded
--- assert that inHeight does not exceed max
*/

/*
Experiments as you scale:
- try one as alpha
-- works the same if not slightly worse at a small scale
- try 0 - 1 param init
-- works quite a bit worse at a small scale
*/

int main()
{
	size_t free, total;
	float* arr;

	cudaDeviceSynchronize();
	cudaMemGetInfo(&free, &total);
	printf("free: %lu\n", free);
	printf("total: %lu\n", total);

	cudaError_t err = cudaMalloc(&arr, free);
	if (err != cudaSuccess)
		printf("CUDA error: %s\n", cudaGetErrorString(err));

	cudaDeviceSynchronize();
	cudaMemGetInfo(&free, &total);
	printf("free: %lu\n", free);
	printf("total: %lu\n", total);

	cudaFree(arr);

	return 0;

	NeuralNetwork nn;
	nn.ExpectInWidth(16);
	nn.ExpectOutWidth(8);
	nn.AddLayer(16, 8, 8, 8);
	nn.AddLayer(16, 1, 8, 1);
	nn.Compile();
	//nn.PrintParameters();

	return 0;
	/*cublasHandle_t cublasHandle;
	cublasCreate(&cublasHandle);

	curandGenerator_t curandGenerator;
	curandCreateGenerator(&curandGenerator, CURAND_RNG_PSEUDO_DEFAULT);
	curandSetPseudoRandomGeneratorSeed(curandGenerator, std::chrono::high_resolution_clock::now().time_since_epoch().count());

	float learningrate = 0.01f;
	int inHeight = 1024, inWidth = 16, hiddenWidth = 16, hiddenHeight = 1, outWidth = 8, heads = 1;
	int outputSize = outWidth * hiddenHeight * heads;
	float* input = new float[inWidth * inHeight];
	float* outputGrad = new float[outputSize * inHeight];

	CLU clu
	(
		&inHeight, inWidth, hiddenWidth, hiddenHeight, outWidth, heads,
		input, outputGrad
	);

	for (int epoch = 0; epoch < 1000; ++epoch)
	{
		for (int i = 0; i < inHeight; ++i)
		{
			uint8_t a = rand();
			uint8_t b = rand();
			uint8_t c = a | b;

			for (int j = 0; j < int(inWidth * 0.5); ++j)
				input[i * inWidth + j] = (a >> j) & 1;
			for (int j = 0; j < int(inWidth * 0.5); ++j)
				input[i * inWidth + j + int(inWidth * 0.5)] = (b >> j) & 1;
			for (int j = 0; j < outputSize; ++j)
				outputGrad[i * outputSize + j] = (c >> j) & 1;
		}

		clu.forward();

		float err = 0;
		for (int i = 0; i < outputSize * inHeight; ++i)
		{
			outputGrad[i] = outputGrad[i] - (clu.output[i] > 0 ? 1 : 0);
			err += abs(outputGrad[i]);
		}

		clu.backward(learningrate);
		err = err / (outputSize * inHeight);
		printf("err: %f\n", err);
	}
	printf("\n");

	clu.printParameters();

	return 0;*/
}