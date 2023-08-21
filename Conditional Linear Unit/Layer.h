#pragma once
#include "GpuMemoryManager.cuh"

struct Layer
{
	virtual ~Layer() = default;
	virtual void Initialize(size_t* inputWidth, float* deviceInputTensor, GpuMemoryManager& gpuMemoryManager) = 0;
	virtual void Forward() = 0;
	virtual void Backward() = 0;
	virtual size_t* GetOutputWidth() = 0;
	virtual float* GetOutputTensor() = 0;
};