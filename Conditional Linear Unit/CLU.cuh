#include "Header.cuh"

struct CLU
{
	int* inHeight, inWidth, hiddenWidth, hiddenHeight, outWidth, heads;

	int nonlinearWidth, jointWidth, productWidth, outputSize, batches;
	float invSqrtInWidth, invsqrtHiddenWidth, invSqrtOutWidth, invSqrtProductWidth, invSqrtInHeight;
	float expDecayMean, expDecayVar;
	float beta1, beta2, epsilon;

	float* input, * weight, * product, * bias, * output;
	float* outputGrad, * productGrad, * biasGrad, * inputGrad, * weightGrad;
	float* weightGradMean, * weightGradVar, * biasGradMean, * biasGradVar;

	static constexpr float one = 1.0f;
	static constexpr float zero = 0.0f;
};