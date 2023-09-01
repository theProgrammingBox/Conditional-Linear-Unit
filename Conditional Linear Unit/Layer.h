#pragma once
#include "GpuMemoryManager.cuh"

struct Layer
{
	// neural network init
	cublasHandle_t* cublasHandle;
	GpuMemoryManager* gpuMemoryManager;
	GpuRand* gpuRand;
	size_t* inputHeight;
	float* learningRate;

	// initialized/shared with layers
	size_t inputWidth, outputWidth;
	float* deviceInputTensor, * deviceOutputTensor;
	float* deviceOutputGradientTensor, * deviceInputGradientTensor;

	virtual void ProvideAllocationDetails() = 0;
	virtual void InitializeParameters() = 0;
	virtual void Forward() = 0;
	virtual void Backward() = 0;
	virtual void PrintParameters() = 0;
};