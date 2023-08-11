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
	float* input, * weight, * product, *bias, *output;
	float* outputGrad, * productGrad, * inputGrad;
	static constexpr float one = 1.0f;
	static constexpr float zero = 0.0f;

	CLU
	(
		float* input, int* inHeight, int inWidth,
		int hiddenWidth, int hiddenHeight, int outWidth,
		float* outputGrad
	) :
		input(input), inHeight(inHeight), inWidth(inWidth),
		hiddenWidth(hiddenWidth), hiddenHeight(hiddenHeight), outWidth(outWidth),
		outputGrad(outputGrad)
	{
		productWidth = hiddenWidth * (hiddenHeight + outWidth);

		hiddenSize = hiddenWidth * hiddenHeight;
		outputSize = outWidth * hiddenHeight;

		// try invsqrt
		invSqrtInWidth = InvSqrt(inWidth);
		invsqrtHiddenWidth = InvSqrt(hiddenWidth);
		invSqrtOutWidth = InvSqrt(outWidth);
		invSqrtProductWidth = InvSqrt(productWidth);
		invSqrtInHeight = InvSqrt(*inHeight);

		// inHeight max calculation, for now, it is static
		weight = new float[productWidth * inWidth];
		product = new float[productWidth * (*inHeight)];
		bias = new float[productWidth];
		output = new float[outputSize * (*inHeight)];

		productGrad = new float[productWidth * (*inHeight)];
		inputGrad = new float[inWidth * (*inHeight)];

		// initialize weight
		for (int i = 0; i < productWidth * inWidth; ++i)
			weight[i] = RandomFloat();
		for (int i = 0; i < productWidth; ++i)
			bias[i] = RandomFloat();
	}

	~CLU()
	{
		delete[] weight;
		delete[] product;
		delete[] bias;
		delete[] output;
		delete[] productGrad;
		delete[] inputGrad;
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

		// add to bias
		float alpha = learningrate * invSqrtInHeight;
		for (int i = 0; i < *inHeight; ++i)
		{
			cpuSaxpy
			(
				productWidth,
				&alpha,
				productGrad + i * productWidth, 1,
				bias, 1
			);
		}

		// binary backward
		/*for (int i = 0; i < *inHeight; ++i)
		{
			cpuBinaryBackward
			(
				hiddenSize,
				&alpha,
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
			&alpha,
			productGrad, productWidth, 0,
			input, inWidth, 0,
			&one,
			weight, productWidth, 0,
			1
		);

		//PrintTensorf32(productWidth, inWidth, weight, "weight");
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