#pragma once
#include "CLU.cuh"

struct NeuralNetwork
{
	cublasHandle_t cublasHandle;
	curandGenerator_t curandGenerator;
	GpuMemoryManager gpuMemoryManager;

	float* hostInputTensor, * hostOutputTensor;
	float* hostOutputGradientTensor, * hostInputGradientTensor;
	float* learningrate;
	size_t* batches, * inputWidth;

	float* deviceInputTensor, * deviceOutputTensor;
	float* deviceOutputGradientTensor, * deviceInputGradientTensor;
	size_t maxBatches;

	std::vector<Layer*> layers;

	NeuralNetwork
	(
		float* hostInputTensor, float* hostOutputTensor,
		float* hostOutputGradientTensor, float* hostInputGradientTensor,
		float* learningrate, size_t* batches
	) :
		hostInputTensor(hostInputTensor), hostOutputTensor(hostOutputTensor),
		hostOutputGradientTensor(hostOutputGradientTensor), hostInputGradientTensor(hostInputGradientTensor),
		learningrate(learningrate), batches(batches)
	{
		cublasStatus_t cublasStatus;
		curandStatus_t curandStatus;

		cublasStatus = cublasCreate(&cublasHandle);
		FailIf(cublasStatus != CUBLAS_STATUS_SUCCESS, "cublasCreate failed");

		curandStatus = curandCreateGenerator(&curandGenerator, CURAND_RNG_PSEUDO_DEFAULT);
		FailIf(curandStatus != CURAND_STATUS_SUCCESS, "curandCreateGenerator failed");

		curandStatus = curandSetPseudoRandomGeneratorSeed(curandGenerator, (uint64_t)time(NULL));
		FailIf(curandStatus != CURAND_STATUS_SUCCESS, "curandSetPseudoRandomGeneratorSeed failed");

		gpuMemoryManager.Init();
	}

	void AddLayer(Layer* layer)
	{
		layer->cublasHandle = &cublasHandle;
		layer->curandGenerator = &curandGenerator;
		layer->learningrate = learningrate;
		layer->batches = batches;
		layers.push_back(layer);
	}

	void Initialize(size_t* inputWidth, size_t* outputWidth)
	{
		FailIf(*outputWidth != layers.back()->outputWidth, "outputWidth != layers.back()->outputWidth");
		
		gpuMemoryManager.ManageDynamic(&deviceInputTensor, *inputWidth);
		layers.front()->Initialize(inputWidth, &gpuMemoryManager);
		for (size_t i = 1; i < layers.size(); i++)
			layers[i]->Initialize(&layers[i - 1]->outputWidth, &gpuMemoryManager);

		gpuMemoryManager.Allocate(maxBatches);
		printf("maxBatches: %d\n\n", maxBatches);
	}

	void Forward()
	{
		FailIf(*batches > maxBatches, "*batches > maxBatches");
	}

	void Backward()
	{
		FailIf(*batches > maxBatches, "*batches > maxBatches");
	}
};