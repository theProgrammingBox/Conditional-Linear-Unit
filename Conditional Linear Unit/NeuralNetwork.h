#pragma once
#include "CLU.cuh"

struct NeuralNetwork
{
	cublasHandle_t cublasHandle;
	GpuMemoryManager gpuMemoryManager;

	float* hostInputTensor, * hostOutputTensor;
	float* hostOutputGradientTensor, * hostInputGradientTensor;
	float* learningrate;
	size_t* batches;
	size_t* inputWidth, * outputWidth;

	float* deviceInputTensor, * deviceOutputTensor;
	float* deviceOutputGradientTensor, * deviceInputGradientTensor;
	size_t maxBatches;

	std::vector<Layer*> layers;

	NeuralNetwork()
	{
		cublasStatus_t cublasStatus;

		cublasStatus = cublasCreate(&cublasHandle);
		FailIf(cublasStatus != CUBLAS_STATUS_SUCCESS, "cublasCreate failed");

		gpuMemoryManager.Init();
	}

	~NeuralNetwork()
	{
		cublasStatus_t cublasStatus;
		cublasStatus = cublasDestroy(cublasHandle);
		FailIf(cublasStatus != CUBLAS_STATUS_SUCCESS, "cublasDestroy failed");
		delete[] hostInputTensor;
		delete[] hostOutputTensor;
		delete[] hostOutputGradientTensor;
		delete[] hostInputGradientTensor;
	}

	void AddLayer(Layer* layer)
	{
		layer->cublasHandle = &cublasHandle;
		layers.push_back(layer);
	}

	void Initialize
	(
		float** hostInputTensor, float** hostOutputTensor,
		float** hostOutputGradientTensor, float** hostInputGradientTensor,
		size_t* inputWidth, size_t* outputWidth
	)
	{
		FailIf(*outputWidth != layers.back()->outputWidth, "outputWidth != layers.back()->outputWidth");

		this->inputWidth = inputWidth;
		this->outputWidth = outputWidth;

		gpuMemoryManager.ManageDynamic(&deviceInputTensor, *inputWidth);
		layers.front()->Initialize(inputWidth, &gpuMemoryManager);
		for (size_t i = 1; i < layers.size(); i++)
			layers[i]->Initialize(&layers[i - 1]->outputWidth, &gpuMemoryManager);

		gpuMemoryManager.Allocate(maxBatches);
		printf("maxBatches: %zu\n\n", maxBatches);

		this->hostInputTensor = new float[*inputWidth * maxBatches];
		this->hostOutputTensor = new float[*outputWidth * maxBatches];
		this->hostOutputGradientTensor = new float[*outputWidth * maxBatches];
		this->hostInputGradientTensor = new float[*inputWidth * maxBatches];

		*hostInputTensor = this->hostInputTensor;
		*hostOutputTensor = this->hostOutputTensor;
		*hostOutputGradientTensor = this->hostOutputGradientTensor;
		*hostInputGradientTensor = this->hostInputGradientTensor;

		deviceInputTensor = layers.front()->deviceInputTensor;
		deviceOutputTensor = layers.back()->deviceOutputTensor;
		deviceOutputGradientTensor = layers.back()->deviceOutputGradientTensor;
		deviceInputGradientTensor = layers.front()->deviceInputGradientTensor;
	}

	void Forward()
	{
		FailIf(*batches > maxBatches, "*batches > maxBatches");

		cudaMemcpy(deviceInputTensor, hostInputTensor, *inputWidth * *batches * sizeof(float), cudaMemcpyHostToDevice);

		/*for (size_t i = 0; i < layers.size(); i++)
			layers[i]->Forward();*/

		//cudaMemcpy(hostOutputTensor, deviceOutputTensor, *outputWidth * *batches * sizeof(float), cudaMemcpyDeviceToHost);
	}

	void Backward()
	{
		FailIf(*batches > maxBatches, "*batches > maxBatches");

		cudaMemcpy(deviceOutputGradientTensor, hostOutputGradientTensor, *outputWidth * *batches * sizeof(float), cudaMemcpyHostToDevice);

		for (size_t i = layers.size() - 1; i < layers.size(); i--)
			layers[i]->Backward();

		cudaMemcpy(hostInputGradientTensor, deviceInputGradientTensor, *inputWidth * *batches * sizeof(float), cudaMemcpyDeviceToHost);
	}

	void PrintParameters()
	{
		for (size_t i = 0; i < layers.size(); i++)
			layers[i]->PrintParameters();
	}
};