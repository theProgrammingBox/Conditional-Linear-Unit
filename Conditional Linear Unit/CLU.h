#pragma once
#include "Layer.cuh"

struct CLU : public Layer
{
	size_t hiddenWidth, hiddenHeight, resultWidth, heads;
	size_t nonlinearWidth, integratedWidth, productWidth, resultSize;

	float* deviceWeightTensor, * deviceBiasTensor;
	float* deviceProductTensor;

	float* deviceWeightGradTensor, * deviceBiasGradTensor;
	float* deviceProductGradTensor;

	float expDecayMean, expDecayVar;
	float beta1, beta2, epsilon;
	float* deviceWeightGradMeanTensor, * deviceWeightGradVarTensor;
	float* deviceBiasGradMeanTensor, * deviceBiasGradVarTensor;

	CLU
	(
		size_t hiddenWidth, size_t hiddenHeight, size_t resultWidth, size_t heads,
		float* learningRate
	) :
		hiddenWidth(hiddenWidth), hiddenHeight(hiddenHeight), resultWidth(resultWidth), heads(heads),
		Layer(learningRate)
	{
		nonlinearWidth = hiddenHeight * hiddenWidth;
		integratedWidth = nonlinearWidth + hiddenWidth * resultWidth;
		productWidth = integratedWidth * heads;
		resultSize = hiddenHeight * resultWidth;
		outputWidth = resultSize * heads;

		expDecayMean = 1.0f;
		expDecayVar = 1.0f;
	}

	void DescribeTensorDetails(size_t* inputWidth, GpuMemoryManager* gpuMemoryManager)
	{
		this->inputWidth = inputWidth;

		gpuMemoryManager->ManageStatic(&deviceWeightTensor, *inputWidth * productWidth);
		gpuMemoryManager->ManageStatic(&deviceBiasTensor, productWidth);

		gpuMemoryManager->ManageDynamic(&deviceProductTensor, productWidth);
		gpuMemoryManager->ManageDynamic(&deviceOutputTensor, outputWidth);

		gpuMemoryManager->ManageStatic(&deviceWeightGradTensor, *inputWidth * productWidth);
		gpuMemoryManager->ManageStatic(&deviceBiasGradTensor, productWidth);

		gpuMemoryManager->ManageDynamic(&deviceProductGradTensor, productWidth);
		gpuMemoryManager->ManageDynamic(&deviceInputGradTensor, *inputWidth);

		gpuMemoryManager->ManageStatic(&deviceWeightGradMeanTensor, *inputWidth * productWidth);
		gpuMemoryManager->ManageStatic(&deviceWeightGradVarTensor, *inputWidth * productWidth);

		gpuMemoryManager->ManageStatic(&deviceBiasGradMeanTensor, productWidth);
		gpuMemoryManager->ManageStatic(&deviceBiasGradVarTensor, productWidth);
	}

	void InitializeParameters(GpuRand* gpuRand)
	{
		gpuRand->Randomize(deviceWeightTensor, *inputWidth * productWidth);
		gpuRand->Randomize(deviceBiasTensor, productWidth);

		cudaMemset(deviceWeightGradMeanTensor, 0, *inputWidth * productWidth * sizeof(float));
		cudaMemset(deviceWeightGradVarTensor, 0, *inputWidth * productWidth * sizeof(float));

		cudaMemset(deviceBiasGradMeanTensor, 0, productWidth * sizeof(float));
		cudaMemset(deviceBiasGradVarTensor, 0, productWidth * sizeof(float));
	}

	void Forward() override
	{
		size_t resultLength = *batches * heads;

		float invSqrtInputWidth = InvSqrt(*inputWidth);
		float invSqrtHiddenWidth = InvSqrt(hiddenWidth);
		float zero = 0.0f;

		cublasSgemm
		(
			*cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N,
			productWidth, *batches, *inputWidth,
			&invSqrtInputWidth,
			deviceWeightTensor, productWidth,
			deviceInputTensor, *inputWidth,
			&zero,
			deviceProductTensor, productWidth
		);

		float* hostInputTensor = new float[*inputWidth * *batches];
		float* hostProductTensor = new float[productWidth * *batches];

		cudaMemcpy(hostInputTensor, deviceInputTensor, *inputWidth * *batches * sizeof(float), cudaMemcpyDeviceToHost);
		cudaMemcpy(hostProductTensor, deviceProductTensor, productWidth * *batches * sizeof(float), cudaMemcpyDeviceToHost);

		PrintTensorf32(*inputWidth, *batches, hostInputTensor, "Input Tensor");
		PrintTensorf32(productWidth, *batches, hostProductTensor, "Product Tensor");

		gpuAdd
		(
			deviceBiasTensor, deviceProductTensor,
			productWidth, *batches
		);

		cudaMemcpy(hostProductTensor, deviceProductTensor, productWidth * *batches * sizeof(float), cudaMemcpyDeviceToHost);

		PrintTensorf32(productWidth, *batches, hostProductTensor, "Product Tensor with bias");

		gpuBinary
		(
			deviceProductTensor,
			nonlinearWidth, *batches, productWidth
		);

		cudaMemcpy(hostProductTensor, deviceProductTensor, productWidth * *batches * sizeof(float), cudaMemcpyDeviceToHost);

		PrintTensorf32(productWidth, *batches, hostProductTensor, "Product Tensor after binary");

		cublasSgemmStridedBatched
		(
			*cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N,
			resultWidth, hiddenHeight, hiddenWidth,
			&invSqrtHiddenWidth,
			deviceProductTensor + nonlinearWidth, resultWidth, integratedWidth,
			deviceProductTensor, hiddenWidth, integratedWidth,
			&zero,
			deviceOutputTensor, resultWidth, resultSize,
			resultLength
		);

		float* hostOutputTensor = new float[outputWidth * *batches];

		cudaMemcpy(hostOutputTensor, deviceOutputTensor, outputWidth * *batches * sizeof(float), cudaMemcpyDeviceToHost);

		PrintTensorf32(outputWidth, *batches, hostOutputTensor, "Output Tensor");
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