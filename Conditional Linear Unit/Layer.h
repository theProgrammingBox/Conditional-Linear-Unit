#pragma once
#include "GpuMemoryManager.cuh"

struct Layer
{
	cublasHandle_t* cublasHandle;
	curandGenerator_t* curandGenerator;

	size_t* batches, * inputWidth, outputWidth;
	float* learningrate;
	float* deviceInputTensor, * deviceOutputTensor;
	float* deviceOutputGradientTensor, * deviceInputGradientTensor;

	virtual void Initialize(size_t* inputWidth, GpuMemoryManager* gpuMemoryManager) = 0;
	virtual void Forward() = 0;
	virtual void Backward() = 0;
};