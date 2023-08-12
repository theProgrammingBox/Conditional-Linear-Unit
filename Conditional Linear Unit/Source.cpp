#include "Header.h"

/*
TODO:
- add adam optimizer
- add multiple "heads"
- think about data layout for future optimizations with recursion
- make inHeight dynamic by calculating the max it can allocate and ensure it is not exceeded
- see if you can make cpuSaxpy and cpuBinaryForward like cpuSgemmStridedBatched
*/

/*
THOUGHTS:
- 
*/

struct CLU
{
	// inHeight can change as it is out batch size
	int* inHeight, inWidth, hiddenWidth, hiddenHeight, outWidth;
	int productWidth, hiddenSize, outputSize;
	float invSqrtInWidth, invsqrtHiddenWidth, invSqrtOutWidth, invSqrtProductWidth, invSqrtInHeight;
	float expDecayMean, expDecayVar;	// for adam optimizer
	float* input, * weight, * product, *bias, *output;
	float* outputGrad, * productGrad, * inputGrad, * weightGrad, * biasGrad;
	float* weightGradMean, * weightGradVar, * biasGradMean, * biasGradVar;	// for adam optimizer
	static constexpr float one = 1.0f;
	static constexpr float zero = 0.0f;
	float beta1, beta2, epsilon;	// for adam optimizer

	CLU
	(
		float* input, int* inHeight, int inWidth,
		int hiddenWidth, int hiddenHeight, int outWidth,
		float* outputGrad,
		float beta1 = 0.9f, float beta2 = 0.999f, float epsilon = 1e-16f
	) :
		input(input), inHeight(inHeight), inWidth(inWidth),
		hiddenWidth(hiddenWidth), hiddenHeight(hiddenHeight), outWidth(outWidth),
		outputGrad(outputGrad),
		beta1(beta1), beta2(beta2), epsilon(epsilon)
	{
		productWidth = hiddenWidth * (hiddenHeight + outWidth);

		hiddenSize = hiddenWidth * hiddenHeight;
		outputSize = outWidth * hiddenHeight;

		invSqrtInWidth = InvSqrt(inWidth);
		invsqrtHiddenWidth = InvSqrt(hiddenWidth);
		invSqrtOutWidth = InvSqrt(outWidth);
		invSqrtProductWidth = InvSqrt(productWidth);
		invSqrtInHeight = InvSqrt(*inHeight);

		expDecayMean = 1.0f;
		expDecayVar = 1.0f;

		// inHeight max calculation, for now, it is static
		weight = new float[productWidth * inWidth];
		product = new float[productWidth * (*inHeight)];
		bias = new float[productWidth];
		output = new float[outputSize * (*inHeight)];

		productGrad = new float[productWidth * (*inHeight)];
		inputGrad = new float[inWidth * (*inHeight)];
		weightGrad = new float[productWidth * inWidth];
		biasGrad = new float[productWidth];

		weightGradMean = new float[productWidth * inWidth];
		weightGradVar = new float[productWidth * inWidth];
		biasGradMean = new float[productWidth];
		biasGradVar = new float[productWidth];

		// initialize params
		for (int i = 0; i < productWidth * inWidth; ++i)
			weight[i] = RandomFloat();
		for (int i = 0; i < productWidth; ++i)
			bias[i] = RandomFloat();

		memset(weightGradMean, 0, productWidth * inWidth * sizeof(float));
		memset(weightGradVar, 0, productWidth * inWidth * sizeof(float));
		memset(biasGradMean, 0, productWidth * sizeof(float));
		memset(biasGradVar, 0, productWidth * sizeof(float));
	}

	~CLU()
	{
		delete[] weight;
		delete[] product;
		delete[] bias;
		delete[] output;

		delete[] productGrad;
		delete[] inputGrad;
		delete[] weightGrad;
		delete[] biasGrad;

		delete[] weightGradMean;
		delete[] weightGradVar;
		delete[] biasGradMean;
		delete[] biasGradVar;
	}

	void forward()
	{
		//PrintTensorf32(inWidth, *inHeight, input, "input");
		//PrintTensorf32(productWidth, inWidth, weight, "weight");

		cpuSgemmStridedBatched
		(
			false, false,
			productWidth, *inHeight, inWidth,
			&invSqrtInWidth,
			weight, productWidth, 0,
			input, inWidth, 0,
			&zero,
			product, productWidth, 0,
			1
		);

		//PrintTensorf32(productWidth, *inHeight, product, "product");
		//PrintTensorf32(productWidth, 1, bias, "bias");

		for (int i = 0; i < *inHeight; ++i)
		{
			cpuSaxpy
			(
				productWidth,
				&one,
				bias, 1,
				product + i * productWidth, 1
			);
		}

		//PrintTensorf32(productWidth, *inHeight, product, "added bias");
		for (int i = 0; i < *inHeight; ++i)
		{
			cpuBinaryForward
			(
				hiddenSize,
				&one,
				product + i * productWidth,
				&zero,
				product + i * productWidth
			);
		}

		//PrintTensorf32(productWidth, *inHeight, product, "full product tensor");
		//PrintTensorf32(hiddenWidth, hiddenHeight, product, "binary forward", 0, productWidth, *inHeight);
		//PrintTensorf32(outWidth, hiddenWidth, product + hiddenSize, "Linear forward", 0, productWidth, *inHeight);
		cpuSgemmStridedBatched
		(
			false, false,
			outWidth, hiddenHeight, hiddenWidth,
			&invsqrtHiddenWidth,
			product + hiddenSize, outWidth, productWidth,
			product, hiddenWidth, productWidth,
			&zero,
			output, outWidth, outputSize,
			*inHeight
		);

		//PrintTensorf32(outputSize, *inHeight, output, "output");
	}

	void backward(float learningrate)
	{
		//PrintTensorf32(outWidth, hiddenHeight, outputGrad, "outputGrad", 0, outputSize, *inHeight);
		cpuSgemmStridedBatched
		(
			true, false,
			hiddenWidth, hiddenHeight, outWidth,
			&invSqrtOutWidth,
			product + hiddenSize, outWidth, productWidth,
			outputGrad, outWidth, outputSize,
			&zero,
			productGrad, hiddenWidth, productWidth,
			*inHeight
		);

		//PrintTensorf32(hiddenWidth, hiddenHeight, productGrad, "binaryGrad", 0, productWidth, *inHeight);
		cpuSgemmStridedBatched
		(
			false, true,
			outWidth, hiddenWidth, hiddenHeight,
			&invsqrtHiddenWidth,
			outputGrad, outWidth, outputSize,
			product, hiddenWidth, productWidth,
			&zero,
			productGrad + hiddenSize, outWidth, productWidth,
			*inHeight
		);

		//PrintTensorf32(outWidth, hiddenWidth, productGrad + hiddenSize, "linearGrad", 0, productWidth, *inHeight);
		//PrintTensorf32(productWidth, *inHeight, productGrad, "productGrad");

		// add to biasGrad
		memset(biasGrad, 0, productWidth * sizeof(float));
		for (int i = 0; i < *inHeight; ++i)
		{
			cpuSaxpy
			(
				productWidth,
				&invSqrtInHeight,
				productGrad + i * productWidth, 1,
				biasGrad, 1
			);
		}

		//PrintTensorf32(productWidth, 1, biasGrad, "biasGrad");
		// binary backward
		/*for (int i = 0; i < *inHeight; ++i)
		{
			cpuBinaryBackward
			(
				hiddenSize,
				&one,
				product + i * productWidth,
				productGrad + i * productWidth,
				product + i * productWidth,
				&zero,
				productGrad + i * productWidth
			);
		}*/

		//PrintTensorf32(productWidth, *inHeight, productGrad, "binaryGrad");
		cpuSgemmStridedBatched
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
		cpuSgemmStridedBatched
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
		
		// apply gradients to parameters using adam
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
	}

	void printParameters() const
	{
		PrintTensorf32(productWidth, inWidth, weight, "weight");
		PrintTensorf32(productWidth, 1, bias, "bias");
	}

	void printWork() const
	{
		PrintTensorf32(inWidth, *inHeight, input, "input");
		PrintTensorf32(hiddenWidth, hiddenHeight, product, "binary forward", 0, productWidth, *inHeight);
		PrintTensorf32(outWidth, hiddenWidth, product + hiddenSize, "Linear forward", 0, productWidth, *inHeight);
		PrintTensorf32(outputSize, *inHeight, output, "output");
	}
};

int main()
{
	srand(time(NULL));

	float learningrate = 0.01f;
	int inHeight = 1024, inWidth = 16, hiddenWidth = 16, hiddenHeight = 2, outWidth = 4;
	int outputSize = outWidth * hiddenHeight;
	float* input = new float[inWidth * inHeight];
	float* outputGrad = new float[outputSize * inHeight];

	CLU clu(input, &inHeight, inWidth, hiddenWidth, hiddenHeight, outWidth, outputGrad);

	for (int epoch = 0; epoch < 100000; ++epoch)
	{
		for (int i = 0; i < inHeight; ++i)
		{
			uint8_t a = rand();
			uint8_t b = rand();
			uint8_t c = a & b;

			for (int j = 0; j < int(inWidth * 0.5); ++j)
				input[i * inWidth + j] = (a >> j) & 1;
			for (int j = 0; j < int(inWidth * 0.5); ++j)
				input[i * inWidth + j + int(inWidth * 0.5)] = (b >> j) & 1;
			for (int j = 0; j < outputSize; ++j)
				outputGrad[i * outputSize + j] = (c >> j) & 1;
		}

		clu.forward();

		float err = 0;
		for (int i = 0; i < outputSize * inHeight; ++i)
		{
			outputGrad[i] = outputGrad[i] - (clu.output[i] > 0 ? 1 : 0);
			err += abs(outputGrad[i]);
		}

		clu.backward(learningrate);
		err = err / (outputSize * inHeight);
		printf("err: %f\n", err);
	}
	printf("\n");

	/*clu.printParameters();
	clu.printWork();*/

	return 0;
}