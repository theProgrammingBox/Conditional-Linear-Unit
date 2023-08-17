#pragma once
#include "Header.cuh"

struct CLU
{
	cublasHandle_t* cublasHandle;
	curandGenerator_t* curandGenerator;

	size_t* inputHeight, * inputWidth;
	size_t hiddenHeight, hiddenWidth, resultWidth, heads;
	size_t nonlinearWidth, integratedWidth, productWidth, resultSize, outputWidth;
	float* learningrate;

	float* input, * weight, * product, * result;

	CLU
	(
		cublasHandle_t* cublasHandle, curandGenerator_t* curandGenerator, float* learningrate,
		size_t* inputHeight, size_t hiddenHeight, size_t hiddenWidth, size_t resultWidth, size_t heads
	) :
		cublasHandle(cublasHandle), curandGenerator(curandGenerator), learningrate(learningrate),
		inputHeight(inputHeight), hiddenHeight(hiddenHeight), hiddenWidth(hiddenWidth), resultWidth(resultWidth), heads(heads)
	{
		nonlinearWidth = hiddenWidth * hiddenHeight;
		integratedWidth = nonlinearWidth + resultWidth * hiddenWidth;
		productWidth = integratedWidth * heads;
		resultSize = resultWidth * hiddenWidth;
		outputWidth = resultSize * heads;
	}

	~CLU()
	{
	}

	void Initialize(size_t* inputWidth, float* input)
	{
		this->inputWidth = inputWidth;
		this->input = input;
		//weight = (float*)malloc(sizeof(float) * productWidth * inputWidth[0]);
	}

	void Forward()
	{
		const float alpha = 1.0f;
		const float beta = 0.0f;

		/*cublasSgemm
		(
			cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N,
			productWidth, *inHeight, *inWidth,
			&alpha,
			weight, productWidth,
			input, inWidth,
			&beta,
			product, productWidth
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
};