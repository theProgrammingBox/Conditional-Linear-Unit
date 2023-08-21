#pragma once
#include "GpuMemoryManager.cuh"

struct CLU
{
	cublasHandle_t* cublasHandle;
	curandGenerator_t* curandGenerator;

	size_t* inputHeight, * inputWidth;
	size_t hiddenHeight, hiddenWidth, resultWidth, heads;
	size_t nonlinearWidth, integratedWidth, productWidth, resultSize, outputWidth;
	float* learningrate;

	float* deviceInputTensor, * deviceWeightTensor, * deviceProductTensor, * deviceResultTensor;

	CLU
	(
		cublasHandle_t* cublasHandle, curandGenerator_t* curandGenerator, float* learningrate, size_t* inputHeight,
		size_t hiddenHeight, size_t hiddenWidth, size_t resultWidth, size_t heads
	) :
		cublasHandle(cublasHandle), curandGenerator(curandGenerator), learningrate(learningrate), inputHeight(inputHeight),
		hiddenHeight(hiddenHeight), hiddenWidth(hiddenWidth), resultWidth(resultWidth), heads(heads)
	{
		nonlinearWidth = hiddenWidth * hiddenHeight;
		integratedWidth = nonlinearWidth + resultWidth * hiddenWidth;
		productWidth = integratedWidth * heads;
		resultSize = resultWidth * hiddenWidth;
		outputWidth = resultSize * heads;
	}

	~CLU()
	{
		cudaFree(deviceWeightTensor);
		cudaFree(deviceProductTensor);
		cudaFree(deviceResultTensor);
	}

	void GatherCoefficients(std::vector<uint32_t>& coefficients)
	{
		coefficients.emplace_back(productWidth);
		coefficients.emplace_back(outputWidth);
	}

	void Initialize(size_t* inputWidth, float* deviceInputTensor)
	{
		this->inputWidth = inputWidth;
		this->deviceInputTensor = deviceInputTensor;

		cudaMalloc((void**)&deviceWeightTensor, productWidth * sizeof(float));
	}

	void Forward()
	{
		/*const float alpha = 1.0f;
		const float beta = 0.0f;

		cublasSgemm
		(
			cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N,
			productWidth, *inHeight, *inWidth,
			&alpha,
			deviceWeightTensor, productWidth,
			deviceInputTensor, inWidth,
			&beta,
			deviceProductTensor, productWidth
		);*/
	}

	void Backward()
	{
		
	}

	void PrintParameters() const
	{
	}

	size_t* GetOutputWidth()
	{
		return &outputWidth;
	}

	float* GetOutputTensor()
	{
		return deviceResultTensor;
	}
};