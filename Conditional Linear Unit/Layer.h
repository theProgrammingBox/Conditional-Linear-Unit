#pragma once
#include "GpuMemoryManager.cuh"

struct Layer
{
	cublasHandle_t* cublasHandle;

	size_t* batches, * inputWidth, outputWidth;
	float* learningrate;
	float* deviceInputTensor, * deviceOutputTensor;
	float* deviceOutputGradientTensor, * deviceInputGradientTensor;

	Layer(float* learningrate) : learningrate(learningrate) {}
	virtual void Initialize(size_t* inputWidth, GpuMemoryManager* gpuMemoryManager) = 0;
	virtual void Forward() = 0;
	virtual void Backward() = 0;
	virtual void PrintParameters() = 0;
};