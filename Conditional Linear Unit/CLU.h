#pragma once
#include "Layer.cuh"

struct CLU : public Layer
{
	size_t hiddenHeight, hiddenWidth, resultWidth, heads;
	size_t nonlinearWidth, integratedWidth, productWidth, resultSize;

	float* deviceWeightTensor, * deviceBiasTensor;
	float* deviceProductTensor;

	CLU(size_t hiddenWidth, size_t hiddenHeight, size_t resultWidth, size_t heads) :
		hiddenWidth(hiddenWidth), hiddenHeight(hiddenHeight), resultWidth(resultWidth), heads(heads)
	{
		nonlinearWidth = hiddenHeight * hiddenWidth;
		integratedWidth = nonlinearWidth + hiddenWidth * resultWidth;
		productWidth = integratedWidth * heads;
		resultSize = hiddenHeight * resultWidth;
		outputWidth = resultSize * heads;
	}

	void Initialize(size_t* inputWidth, GpuMemoryManager* gpuMemoryManager)
	{
		this->inputWidth = inputWidth;

		gpuMemoryManager->ManageStatic(&deviceWeightTensor, *inputWidth * productWidth);
		gpuMemoryManager->ManageStatic(&deviceBiasTensor, productWidth);

		gpuMemoryManager->ManageDynamic(&deviceProductTensor, productWidth);
		gpuMemoryManager->ManageDynamic(&deviceOutputTensor, outputWidth);
	}

	void Forward() override
	{
		size_t resultLength = *batches * heads;
	}

	void Backward() override
	{
		size_t resultLength = *batches * heads;
	}
};