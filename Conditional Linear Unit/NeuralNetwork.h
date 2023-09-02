#pragma once
#include "CLU.cuh"

// allow user to pass device tensors
// allow user to choose to copy device tensors to host

struct NeuralNetwork
{
	// passed to all layers as pointers
	cublasHandle_t cublasHandle;
	GpuMemoryManager gpuMemoryManager;
	GpuRand gpuRand;
	size_t inputHeight;
	float learningRate;

	// initialized/shared with layers
	size_t inputWidth, outputWidth;
	float* deviceInputTensor, * deviceOutputTensor;
	float* deviceOutputGradientTensor, * deviceInputGradientTensor;

	// initialized/shared with user
	size_t maxInputHeight;
	float* hostInputTensor, * hostOutputTensor;
	float* hostOutputGradientTensor, * hostInputGradientTensor;

	std::vector<Layer*> layers;

	NeuralNetwork()
	{
		cublasStatus_t cublasStatus;

		cublasStatus = cublasCreate(&cublasHandle);
		FailIf(cublasStatus != CUBLAS_STATUS_SUCCESS, "cublasCreate failed");

		gpuMemoryManager.MapGpuMemory();
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
		layer->gpuMemoryManager = &gpuMemoryManager;
		layer->gpuRand = &gpuRand;
		layer->inputHeight = &inputHeight;
		layer->learningRate = &learningRate;

		layers.emplace_back(layer);
	}

	void Initialize
	(
		size_t inputWidth, size_t outputWidth,
		float*& hostInputTensor, float*& hostOutputTensor,
		float*& hostOutputGradientTensor, float*& hostInputGradientTensor
	)
	{
		FailIf(outputWidth != layers.back()->outputWidth, "outputWidth != layers.back()->outputWidth");

		this->inputWidth = inputWidth;
		this->outputWidth = outputWidth;

		// connect layer dimensions and describe tensor details to memory manager
		gpuMemoryManager.ManageDynamic(&deviceInputTensor, inputWidth);
		layers.front()->inputWidth = inputWidth;
		layers.front()->DescribeTensorDetails();
		for (size_t i = 1; i < layers.size(); i++)
		{
			layers[i]->inputWidth = layers[i - 1]->outputWidth;
			layers[i]->DescribeTensorDetails();
		}

		// memory manager allocates static and dynamic memory
		gpuMemoryManager.Allocate(maxInputHeight);
		printf("maxInputHeight: %zu\n\n", maxInputHeight);

		// connect layer tensors and initialize parameters
		layers.front()->deviceInputTensor = deviceInputTensor;
		layers.front()->InitializeParameters();
		for (size_t i = 1; i < layers.size(); i++)
		{
			layers[i]->deviceInputTensor = layers[i - 1]->deviceOutputTensor;
			layers[i]->InitializeParameters();
		}

		// initialize host tensors
		this->hostInputTensor = new float[*inputWidth * maxInputHeight];
		this->hostOutputTensor = new float[*outputWidth * maxInputHeight];
		this->hostOutputGradientTensor = new float[*outputWidth * maxInputHeight];
		this->hostInputGradientTensor = new float[*inputWidth * maxInputHeight];

		// give host tensors to user
		hostInputTensor = this->hostInputTensor;
		hostOutputTensor = this->hostOutputTensor;
		hostOutputGradientTensor = this->hostOutputGradientTensor;
		hostInputGradientTensor = this->hostInputGradientTensor;

		// set interface tensors for 
		deviceInputTensor = layers.front()->deviceInputTensor;
		deviceOutputTensor = layers.back()->deviceOutputTensor;
		deviceOutputGradientTensor = layers.back()->deviceOutputGradientTensor;
		deviceInputGradientTensor = layers.front()->deviceInputGradientTensor;
	}

	void Forward(size_t inputHeight)
	{
		this->inputHeight = inputHeight;
		FailIf(inputHeight > maxInputHeight, "inputHeight > maxInputHeight");
		cudaMemcpy(deviceInputTensor, hostInputTensor, inputWidth * inputHeight * sizeof(float), cudaMemcpyHostToDevice);

		for (size_t i = 0; i < layers.size(); i++)
			layers[i]->Forward();
	}

	void Backward(float learningRate)
	{
		this->learningRate = learningRate;
		FailIf(inputHeight > maxInputHeight, "inputHeight > maxInputHeight");
		cudaMemcpy(deviceOutputGradientTensor, hostOutputGradientTensor, outputWidth * inputHeight * sizeof(float), cudaMemcpyHostToDevice);

		for (size_t i = layers.size() - 1; i < layers.size(); i--)
			layers[i]->Backward();
	}

	void PrintParameters()
	{
		for (size_t i = 0; i < layers.size(); i++)
			layers[i]->PrintParameters();
	}

	void Planning()
	{
		/*
		hostInput = nullptr;
		deviceInput = deviceInput;
		if (deviceInput == nullptr)
		{
			add dev to manager
			init host
		}

		// connect stuff

		if (hostInput != nullptr)
		{
			copy to dev
		}
		*/
	}

	float* GetHostOutputTensor()
	{
		cudaMemcpy(hostOutputTensor, deviceOutputTensor, outputWidth * inputHeight * sizeof(float), cudaMemcpyDeviceToHost);
		return hostOutputTensor;
	}

	float* GetHostInputGradientTensor()
	{
		cudaMemcpy(hostInputGradientTensor, deviceInputGradientTensor, inputWidth * inputHeight * sizeof(float), cudaMemcpyDeviceToHost);
		return hostInputGradientTensor;
	}
};