#pragma once
#include <iostream>
#include <vector>
#include <cassert>

float InvSqrt(float number)
{
	long i = 0x5F1FFFF9 - (*(long*)&number >> 1);
	float tmp = *(float*)&i;
	return tmp * 0.703952253f * (2.38924456f - number * tmp * tmp);
}

void PrintTensorf32(uint32_t width, uint32_t height, float* arr, const char* label = "Tensor", uint32_t majorStride = 0, uint32_t tensorSize = 0, uint32_t batchCount = 1)
{
	if (majorStride == 0)
		majorStride = width;
	printf("%s:\n", label);
	for (int b = batchCount; b--;)
	{
		for (uint32_t i = 0; i < height; i++)
		{
			for (uint32_t j = 0; j < width; j++)
				printf("%6.3f ", arr[i * majorStride + j]);
			printf("\n");
		}
		printf("\n");
		arr += tensorSize;
	}
}

float RandomFloat()
{
	return rand() * 0.00006103515625f - 1.0f;
}

float RandomGaussian(float mean, float stddev)
{
	float x1, x2, w, y1;
	static float y2;
	static int use_last = 0;

	if (use_last)
	{
		y1 = y2;
		use_last = 0;
	}
	else
	{
		do
		{
			x1 = RandomFloat();
			x2 = RandomFloat();
			w = x1 * x1 + x2 * x2;
		} while (w >= 1.0f);

		w = sqrt((-2.0f * log(w)) / w);
		y1 = x1 * w;
		y2 = x2 * w;
		use_last = 1;
	}

	return (mean + y1 * stddev);
}

void cpuSgemmStridedBatched(
	bool transW, bool transX,
	// determines the length of dot products
	int outWidth, int inHeight, int sharedDim,
	const float* alpha,
	// determines the number of elements to skip to get to the next row/column and tensor
	float* W, int WMajorStride, int WTensorStride,
	float* X, int XMajorStride, int XTensorStride,
	const float* beta,
	float* Y, int YMajorStride, int YTensorStride,
	int batchCount)
{
	for (int b = batchCount; b--;)
	{
		for (int m = outWidth; m--;)
		{
			for (int n = inHeight; n--;)
			{
				float sum = 0;
				for (int k = sharedDim; k--;)
				{
					float a_value = transX ? X[k * XMajorStride + n] : X[n * XMajorStride + k];
					float b_value = transW ? W[m * WMajorStride + k] : W[k * WMajorStride + m];
					sum += a_value * b_value;
				}
				Y[n * YMajorStride + m] = *alpha * sum + *beta * Y[n * YMajorStride + m];
			}
		}
		X += XTensorStride;
		W += WTensorStride;
		Y += YTensorStride;
	}
}

void cpuSaxpy(
	int n,
	const float* alpha,
	const float* x, int incx,
	float* y, int incy)
{
	for (int i = 0; i < n; i++)
		y[i * incy] = *alpha * x[i * incx] + y[i * incy];
}

void cpuBinaryForward(
	int n,
	const float* alpha,
	const float* x,
	const float* beta,
	float* y)
{
	for (int i = 0; i < n; i++)
		y[i] = *beta * y[i] + *alpha * x[i] > 0;
}

void cpuReluForward(
	int n,
	const float* alpha,
	const float* x,
	const float* beta,
	float* y)
{
	for (int i = 0; i < n; i++)
		y[i] = *beta * y[i] + (*alpha * x[i] >= 0 ? *alpha * x[i] : 0);
}

void cpuReluBackward(
	int n,
	const float* alpha,
	const float* y,
	const float* dy,
	const float* x,
	const float* beta,
	float* dx)
{
	for (int i = 0; i < n; i++)
		dx[i] = *beta * dx[i] + (*alpha * x[i] >= 0 ? *alpha * dy[i] : 0);
}

void cpuLeakyReluForward(
	int n,
	const float* alpha,
	const float* x,
	const float* beta,
	float* y)
{
	for (int i = 0; i < n; i++)
		y[i] = *beta * y[i] + (*alpha * x[i] >= 0 ? *alpha * x[i] : 0.1 * *alpha * x[i]);
}

void cpuLeakyReluBackward(
	int n,
	const float* alpha,
	const float* y,
	const float* dy,
	const float* x,
	const float* beta,
	float* dx)
{
	for (int i = 0; i < n; i++)
		dx[i] = *beta * dx[i] + (*alpha * x[i] >= 0 ? *alpha * dy[i] : 0.1 * *alpha * dy[i]);
}

void cpuGeluForward(
	int n,
	const float* alpha,
	const float* x,
	const float* beta,
	float* y)
{
	for (int i = 0; i < n; i++)
	{
		float gelu = x[i] / (1 + exp(-1.702 * x[i]));
		y[i] = *beta * y[i] + *alpha * gelu;
	}
}

void cpuGeluBackward(
	int n,
	const float* alpha,
	const float* y,
	const float* dy,
	const float* x,
	const float* beta,
	float* dx)
{
	for (int i = 0; i < n; i++)
	{
		float z0 = 1.702 * x[i];
		float z1 = exp(z0);
		float z2 = z1 + 1;
		float geluGrad = z1 * (z0 + z1 + 1) / (z2 * z2);
		dx[i] = *beta * dx[i] + *alpha * geluGrad * dy[i];
	}
}

void cpuSigmoidForward(
	int n,
	const float* alpha,
	const float* x,
	const float* beta,
	float* y)
{
	for (int i = 0; i < n; i++)
	{
		float sigmoid = 1 / (1 + exp(-x[i]));
		y[i] = *beta * y[i] + *alpha * sigmoid;
	}
}

void cpuSigmoidBackward(
	int n,
	const float* alpha,
	const float* y,
	const float* dy,
	const float* x,
	const float* beta,
	float* dx)
{
	for (int i = 0; i < n; i++)
	{
		float sigmoid = 1 / (1 + exp(-x[i]));
		float sigmoidGrad = sigmoid * (1 - sigmoid);
		dx[i] = *beta * dx[i] + *alpha * sigmoidGrad * dy[i];
	}
}