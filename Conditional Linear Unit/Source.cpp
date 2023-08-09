#include "Header.h"

/*
TODO:
- make inHeight dynamic by calculating the max it can allocate and ensure it is not exceeded
- see if you can make cpuSaxpy and cpuBinaryForward like cpuSgemmStridedBatched
*/

struct CLU
{
	// inHeight can change as it is out batch size
	int* inHeight, inWidth, hiddenWidth, hiddenHeight, outWidth;
	int productWidth, hiddenSize, outputSize;
	float invInWidth, invHiddenWidth;
	float* input, * weight, * product, *bias, *output;
	static constexpr float one = 1.0f;
	static constexpr float zero = 0.0f;

	CLU
	(
		float* input, int* inHeight, int inWidth,
		int hiddenWidth, int hiddenHeight, int outWidth
	) :
		input(input), inHeight(inHeight), inWidth(inWidth),
		hiddenWidth(hiddenWidth), hiddenHeight(hiddenHeight), outWidth(outWidth)
	{
		productWidth = hiddenWidth * (hiddenHeight * outWidth);

		hiddenSize = hiddenWidth * hiddenHeight;
		outputSize = outWidth * hiddenHeight;

		invInWidth = 1.0f / inWidth;
		invHiddenWidth = 1.0f / hiddenWidth;

		// inHeight max calculation, for now, it is static
		weight = new float[productWidth * inWidth];
		product = new float[productWidth * (*inHeight)];
		bias = new float[productWidth];
		output = new float[outputSize * (*inHeight)];

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
	}

	void forward()
	{
		PrintTensorf32(inWidth, *inHeight, input, "input");
		PrintTensorf32(productWidth, inWidth, weight, "weight");

		cpuSgemmStridedBatched
		(
			false, false,
			productWidth, *inHeight, inWidth,
			&invInWidth,
			weight, productWidth, 0,
			input, inWidth, 0,
			&zero,
			product, productWidth, 0,
			1
		);

		PrintTensorf32(productWidth, *inHeight, product, "product");
		PrintTensorf32(productWidth, 1, bias, "bias");

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

		PrintTensorf32(productWidth, *inHeight, product, "added bias");
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

		PrintTensorf32(productWidth, *inHeight, product, "full product tensor");
		PrintTensorf32(hiddenWidth, hiddenHeight, product, "binary forward", 0, productWidth, *inHeight);
		PrintTensorf32(outWidth, hiddenWidth, product + hiddenSize, "Linear forward", 0, productWidth, *inHeight);
		cpuSgemmStridedBatched
		(
			false, false,
			outWidth, hiddenHeight, hiddenWidth,
			&invHiddenWidth,
			product + hiddenSize, outWidth, productWidth,
			product, hiddenWidth, productWidth,
			&zero,
			output, outWidth, outputSize,
			*inHeight
		);

		PrintTensorf32(outWidth, hiddenHeight, output, "output", 0, outWidth, *inHeight);
	}
};

int main()
{
	int inHeight = 6, inWidth = 5, hiddenWidth = 2, hiddenHeight = 3, outWidth = 4;
	float* input = new float[inWidth * inHeight];
	for (int i = 0; i < inWidth * inHeight; ++i)
		input[i] = RandomFloat();

	CLU clu(input, &inHeight, inWidth, hiddenWidth, hiddenHeight, outWidth);
	clu.forward();

	return 0;
}