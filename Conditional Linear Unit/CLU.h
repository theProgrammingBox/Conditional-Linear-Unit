#pragma once
#include "Layer.cuh"

struct CLU : public Layer
{
	size_t hiddenHeight, hiddenWidth, resultWidth, heads;
	size_t nonlinearWidth, integratedWidth, productWidth, resultSize;

	float* deviceWeightTensor, * deviceBiasTensor;
	float* deviceProductTensor;

	CLU
	(
		size_t hiddenWidth, size_t hiddenHeight, size_t resultWidth, size_t heads,
		float* learningrate
	) :
		hiddenWidth(hiddenWidth), hiddenHeight(hiddenHeight), resultWidth(resultWidth), heads(heads),
		Layer(learningrate)
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

		const float alpha = 1.0f;
		const float beta = 0.0f;

		cublasSgemm
		(
			*cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N,
			productWidth, *batches, *inputWidth,
			&alpha,
			deviceWeightTensor, productWidth,
			deviceInputTensor, *inputWidth,
			&beta,
			deviceProductTensor, productWidth
		);

		float* hostInputTensor = new float[*inputWidth * *batches];
		float* hostProductTensor = new float[productWidth * *batches];

		cudaMemcpy(hostInputTensor, deviceInputTensor, *inputWidth * *batches * sizeof(float), cudaMemcpyDeviceToHost);
		cudaMemcpy(hostProductTensor, deviceProductTensor, productWidth * *batches * sizeof(float), cudaMemcpyDeviceToHost);

		PrintTensorf32(*inputWidth, *batches, hostInputTensor, "Input Tensor");
		PrintTensorf32(productWidth, *batches, hostProductTensor, "Product Tensor");
	}

	void Backward() override
	{
		size_t resultLength = *batches * heads;
	}

	void PrintParameters()
	{
		float* weightTensor = new float[*inputWidth * productWidth];
		float* biasTensor = new float[productWidth];

		cudaMemcpy(weightTensor, deviceWeightTensor, *inputWidth * productWidth * sizeof(float), cudaMemcpyDeviceToHost);
		cudaMemcpy(biasTensor, deviceBiasTensor, productWidth * sizeof(float), cudaMemcpyDeviceToHost);

		PrintTensorf32(productWidth, *inputWidth, weightTensor, "Weight Tensor");
		PrintTensorf32(productWidth, 1, biasTensor, "Bias Tensor");
	}
};