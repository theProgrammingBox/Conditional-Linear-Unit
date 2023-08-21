#pragma once
#include "CLU.cuh"

struct NeuralNetwork
{
	cublasHandle_t cublasHandle;		// internally initialized
	curandGenerator_t curandGenerator;	// internally initialized // TODO: remove this, use custom

	size_t maxBatches;		// internally initialized
	size_t inputWidth;
	size_t* batches;		// externally altered
	float* learningrate;	// externally altered

	float* deviceInputTensor, * deviceOutputTensor;					// internally initialized and altered
	float* deviceOutputGradientTensor, * deviceInputGradientTensor;	// internally initialized and altered

	float* hostInputTensor, * hostOutputTensor;					// internally initialized and externally altered
	float* hostOutputGradientTensor, * hostInputGradientTensor;	// internally initialized and externally altered

	std::vector<CLU*> layers;

	NeuralNetwork
	(
		float* learningrate, size_t* batches,
		float* hostInputTensor, float* hostOutputTensor,
		float* hostOutputGradientTensor, float* hostInputGradientTensor
	) :
		learningrate(learningrate), batches(batches),
		hostInputTensor(hostInputTensor), hostOutputTensor(hostOutputTensor),
		hostOutputGradientTensor(hostOutputGradientTensor), hostInputGradientTensor(hostInputGradientTensor)
	{
		cublasStatus_t cublasStatus;
		curandStatus_t curandStatus;

		cublasStatus = cublasCreate(&cublasHandle);
		if (cublasStatus != CUBLAS_STATUS_SUCCESS)
			printf("cublasCreate failed with error code %d\n", cublasStatus);

		curandStatus = curandCreateGenerator(&curandGenerator, CURAND_RNG_PSEUDO_DEFAULT);
		if (curandStatus != CURAND_STATUS_SUCCESS)
			printf("curandCreateGenerator failed with error code %d\n", curandStatus);

		curandStatus = curandSetPseudoRandomGeneratorSeed(curandGenerator, (unsigned long long)time(NULL));
		if (curandStatus != CURAND_STATUS_SUCCESS)
			printf("curandSetPseudoRandomGeneratorSeed failed with error code %d\n", curandStatus);
	}

	void AddLayer(size_t hiddenHeight, size_t hiddenWidth, size_t resultWidth, size_t heads)
	{
		layers.push_back
		(
			new CLU
			(
				&cublasHandle, &curandGenerator, learningrate, batches,
				hiddenHeight, hiddenWidth, resultWidth, heads
			)
		);
	}

	void Initialize(size_t inputWidth, size_t outputWidth)
	{
		this->inputWidth = inputWidth;
		assert(outputWidth == *layers.back()->GetOutputWidth());

		GpuMemoryManager gpuMemoryManager;

		gpuMemoryManager.Manage(&deviceInputTensor, inputWidth);

		layers.front()->Initialize(&(this->inputWidth), deviceInputTensor, gpuMemoryManager);
		for (size_t i = 1; i < layers.size(); i++)
			layers[i]->Initialize(layers[i - 1]->GetOutputWidth(), layers[i - 1]->GetOutputTensor(), gpuMemoryManager);

		gpuMemoryManager.PrintGpuMem();
	}

	void Forward()
	{
		assert(*batches <= maxBatches);

		for (size_t i = 0; i < layers.size(); i++)
			layers[i]->Forward();
	}
};