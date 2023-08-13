#pragma once
#include "Header.cuh"

struct CLU
{
	cublasHandle_t* cublasHandle;
	curandGenerator_t* curandGenerator;

	int inWidth, hiddenWidth, hiddenHeight, outWidth, heads;

	int nonlinearWidth, jointWidth, productWidth, outputSize, batches;
	float expDecayMean, expDecayVar;
	float beta1, beta2, epsilon;

	float* input, * weight, * product, * bias, * output;
	float* outputGrad, * productGrad, * biasGrad, * inputGrad, * weightGrad;
	float* weightGradMean, * weightGradVar, * biasGradMean, * biasGradVar;

	static constexpr float one = 1.0f;
	static constexpr float zero = 0.0f;

	CLU
	(
		cublasHandle_t* cublasHandle, curandGenerator_t* curandGenerator,
		int hiddenWidth, int hiddenHeight, int outWidth, int heads,
		float beta1 = 0.9f, float beta2 = 0.999f, float epsilon = 1e-16f
	) :
		cublasHandle(cublasHandle), curandGenerator(curandGenerator),
		hiddenWidth(hiddenWidth), hiddenHeight(hiddenHeight),
		outWidth(outWidth), heads(heads),
		beta1(beta1), beta2(beta2), epsilon(epsilon)
	{
		expDecayMean = 1.0f;
		expDecayVar = 1.0f;
	}

	~CLU()
	{
		cudaFree(input);
		cudaFree(weight);
		cudaFree(product);
		cudaFree(bias);
		cudaFree(output);

		cudaFree(outputGrad);
		cudaFree(productGrad);
		cudaFree(biasGrad);
		cudaFree(inputGrad);
		cudaFree(weightGrad);

		cudaFree(weightGradMean);
		cudaFree(weightGradVar);
		cudaFree(biasGradMean);
		cudaFree(biasGradVar);
	}

	int GetSizeCoefficient()
	{
		nonlinearWidth = hiddenWidth * hiddenHeight;
		jointWidth = nonlinearWidth + outWidth * hiddenWidth;
		productWidth = jointWidth * heads;
		outputSize = outWidth * hiddenHeight;
		return 2 * productWidth + inWidth + outputSize * heads;
	}

	void Allocate(int maxInHeight)
	{
		cudaError_t err;
		/*cudaMalloc(&input, inWidth * maxInHeight * sizeof(float));
		cudaMalloc(&weight, productWidth * inWidth * sizeof(float));
		cudaMalloc(&product, productWidth * maxInHeight * sizeof(float));
		cudaMalloc(&bias, productWidth * sizeof(float));
		cudaMalloc(&output, outputSize * heads * maxInHeight * sizeof(float));*/
		ErrCheckCudaMalloc((void**)&input, inWidth * maxInHeight * sizeof(float), err);
		ErrCheckCudaMalloc((void**)&weight, productWidth * inWidth * sizeof(float), err);
		ErrCheckCudaMalloc((void**)&product, productWidth * maxInHeight * sizeof(float), err);
		ErrCheckCudaMalloc((void**)&bias, productWidth * sizeof(float), err);
		ErrCheckCudaMalloc((void**)&output, outputSize * heads * maxInHeight * sizeof(float), err);
		
		/*cudaMalloc(&outputGrad, outputSize * heads * maxInHeight * sizeof(float));
		cudaMalloc(&productGrad, productWidth * maxInHeight * sizeof(float));
		cudaMalloc(&biasGrad, productWidth * sizeof(float));
		cudaMalloc(&inputGrad, inWidth * maxInHeight * sizeof(float));
		cudaMalloc(&weightGrad, productWidth * inWidth * sizeof(float));*/
		ErrCheckCudaMalloc((void**)&outputGrad, outputSize * heads * maxInHeight * sizeof(float), err);
		ErrCheckCudaMalloc((void**)&productGrad, productWidth * maxInHeight * sizeof(float), err);
		ErrCheckCudaMalloc((void**)&biasGrad, productWidth * sizeof(float), err);
		ErrCheckCudaMalloc((void**)&inputGrad, inWidth * maxInHeight * sizeof(float), err);
		ErrCheckCudaMalloc((void**)&weightGrad, productWidth * inWidth * sizeof(float), err);

		/*cudaMalloc(&weightGradMean, productWidth * inWidth * sizeof(float));
		cudaMalloc(&weightGradVar, productWidth * inWidth * sizeof(float));
		cudaMalloc(&biasGradMean, productWidth * sizeof(float));
		cudaMalloc(&biasGradVar, productWidth * sizeof(float));*/
		ErrCheckCudaMalloc((void**)&weightGradMean, productWidth * inWidth * sizeof(float), err);
		ErrCheckCudaMalloc((void**)&weightGradVar, productWidth * inWidth * sizeof(float), err);
		ErrCheckCudaMalloc((void**)&biasGradMean, productWidth * sizeof(float), err);
		ErrCheckCudaMalloc((void**)&biasGradVar, productWidth * sizeof(float), err);

		CurandGenerateUniformf32(*curandGenerator, weight, productWidth * inWidth);
		cudaMemset(bias, 0, productWidth * sizeof(float));
		cudaMemset(weightGradMean, 0, productWidth * inWidth * sizeof(float));
		cudaMemset(weightGradVar, 0, productWidth * inWidth * sizeof(float));
		cudaMemset(biasGradMean, 0, productWidth * sizeof(float));
		cudaMemset(biasGradVar, 0, productWidth * sizeof(float));
	}

	/*void Forward(int inHeight)
	{
		batches = heads * inHeight;
		invSqrtInHeight = InvSqrt(inHeight);

		//PrintTensorf32(inWidth, *inHeight, input, "input");
		//PrintTensorf32(productWidth, inWidth, weight, "weight");

		cublasSgemmStridedBatched
		(
			false, false,
			productWidth, inHeight, inWidth,
			&invSqrtInWidth,
			weight, productWidth, 0,
			input, inWidth, 0,
			&zero,
			product, productWidth, 0,
			1
		);
		//PrintTensorf32(productWidth, *inHeight, product, "product");
		//PrintTensorf32(productWidth, 1, bias, "bias");

		for (int i = 0; i < inHeight; ++i)
		{
			cublasSaxpy
			(
				productWidth,
				&one,
				bias, 1,
				product + i * productWidth, 1
			);
		}
		//PrintTensorf32(productWidth, *inHeight, product, "added bias");

		for (int i = 0; i < batches; ++i)
		{
			cpuBinaryForward
			(
				nonlinearWidth,
				&one,
				product + i * jointWidth,
				&zero,
				product + i * jointWidth
			);
		}
		//PrintTensorf32(productWidth, *inHeight, product, "full product tensor");
		//PrintTensorf32(hiddenWidth, hiddenHeight, product, "binary forward", 0, productWidth, *inHeight);
		//PrintTensorf32(outWidth, hiddenWidth, product + nonlinearWidth, "Linear forward", 0, productWidth, *inHeight);

		cublasSgemmStridedBatched
		(
			false, false,
			outWidth, hiddenHeight, hiddenWidth,
			&invsqrtHiddenWidth,
			product + nonlinearWidth, outWidth, jointWidth,
			product, hiddenWidth, jointWidth,
			&zero,
			output, outWidth, outputSize,
			batches
		);
		//PrintTensorf32(outputSize, *inHeight, output, "output");
	}

	void Backward(float learningrate)
	{
		//PrintTensorf32(outWidth, hiddenHeight, outputGrad, "outputGrad", 0, outputSize, *inHeight);

		cublasSgemmStridedBatched
		(
			true, false,
			hiddenWidth, hiddenHeight, outWidth,
			&invSqrtOutWidth,
			product + nonlinearWidth, outWidth, jointWidth,
			outputGrad, outWidth, outputSize,
			&zero,
			productGrad, hiddenWidth, jointWidth,
			batches
		);
		//PrintTensorf32(hiddenWidth, hiddenHeight, productGrad, "binaryGrad", 0, productWidth, *inHeight);

		cublasSgemmStridedBatched
		(
			false, true,
			outWidth, hiddenWidth, hiddenHeight,
			&invsqrtHiddenWidth,
			outputGrad, outWidth, outputSize,
			product, hiddenWidth, jointWidth,
			&zero,
			productGrad + nonlinearWidth, outWidth, jointWidth,
			batches
		);
		//PrintTensorf32(outWidth, hiddenWidth, productGrad + nonlinearWidth, "linearGrad", 0, productWidth, *inHeight);
		//PrintTensorf32(productWidth, *inHeight, productGrad, "productGrad");

		memset(biasGrad, 0, productWidth * sizeof(float));
		for (int i = 0; i < *inHeight; ++i)
		{
			cublasSaxpy
			(
				productWidth,
				&invSqrtInHeight,
				productGrad + i * productWidth, 1,
				biasGrad, 1
			);
		}
		//PrintTensorf32(productWidth, 1, biasGrad, "biasGrad");

		cublasSgemmStridedBatched
		(
			true, false,
			inWidth, *inHeight, productWidth,
			&invSqrtProductWidth,
			weight, productWidth, 0,
			productGrad, productWidth, 0,
			&zero,
			inputGrad, inWidth, 0,
			1
		);
		//PrintTensorf32(inWidth, *inHeight, inputGrad, "inputGrad");

		cublasSgemmStridedBatched
		(
			false, true,
			productWidth, inWidth, *inHeight,
			&invSqrtInHeight,
			productGrad, productWidth, 0,
			input, inWidth, 0,
			&zero,
			weightGrad, productWidth, 0,
			1
		);
		//PrintTensorf32(productWidth, inWidth, weightGrad, "weightGrad");

		expDecayMean *= beta1;
		expDecayVar *= beta2;

		for (int i = 0; i < productWidth; ++i)
		{
			float gradient = biasGrad[i];
			float newGradMean = beta1 * biasGradMean[i] + (1.0f - beta1) * gradient;
			float newGradVar = beta2 * biasGradVar[i] + (1.0f - beta2) * gradient * gradient;
			biasGradMean[i] = newGradMean;
			biasGradVar[i] = newGradVar;
			float gradMeanCorrected = newGradMean / (1.0f - expDecayMean);
			float gradVarCorrected = newGradVar / (1.0f - expDecayVar);
			float finalGradient = gradMeanCorrected * InvSqrt(gradVarCorrected + epsilon);
			bias[i] += finalGradient * learningrate;
		}

		for (int i = 0; i < productWidth * inWidth; ++i)
		{
			float gradient = weightGrad[i];
			float newGradMean = beta1 * weightGradMean[i] + (1.0f - beta1) * gradient;
			float newGradVar = beta2 * weightGradVar[i] + (1.0f - beta2) * gradient * gradient;
			weightGradMean[i] = newGradMean;
			weightGradVar[i] = newGradVar;
			float gradMeanCorrected = newGradMean / (1.0f - expDecayMean);
			float gradVarCorrected = newGradVar / (1.0f - expDecayVar);
			float finalGradient = gradMeanCorrected * InvSqrt(gradVarCorrected + epsilon);
			weight[i] += finalGradient * learningrate;
		}
	}*/

	void PrintParameters() const
	{
		/*PrintTensorf32(productWidth, inWidth, weight, "weight");
		PrintTensorf32(productWidth, 1, bias, "bias");*/

		float* weight = new float[productWidth * inWidth];
		float* bias = new float[productWidth];
		cudaMemcpy(weight, this->weight, productWidth * inWidth * sizeof(float), cudaMemcpyDeviceToHost);
		cudaMemcpy(bias, this->bias, productWidth * sizeof(float), cudaMemcpyDeviceToHost);
		PrintTensorf32(productWidth, inWidth, weight, "weight");
		PrintTensorf32(productWidth, 1, bias, "bias");
	}

	int GetInputWidth() const
	{
		return inWidth;
	}

	int GetOutputWidth() const
	{
		return outputSize * heads;
	}
};