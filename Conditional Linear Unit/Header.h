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

void PrintMatrixf32(float* arr, uint32_t height, uint32_t width, const char* label)
{
	printf("%s:\n", label);
	for (uint32_t i = 0; i < height; i++)
	{
		for (uint32_t j = 0; j < width; j++)
			printf("%6.3f ", arr[i * width + j]);
		printf("\n");
	}
	printf("\n");
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
	bool transB, bool transA,
	int CCols, int CRows, int AColsBRows,
	const float* alpha,
	float* B, int ColsB, int SizeB,
	float* A, int ColsA, int SizeA,
	const float* beta,
	float* C, int ColsC, int SizeC,
	int batchCount)
{
	for (int b = batchCount; b--;)
	{
		for (int m = CCols; m--;)
		{
			for (int n = CRows; n--;)
			{
				float sum = 0;
				for (int k = AColsBRows; k--;)
				{
					float a_value = transA ? A[k * ColsA + n] : A[n * ColsA + k];
					float b_value = transB ? B[m * ColsB + k] : B[k * ColsB + m];
					sum += a_value * b_value;
				}
				C[n * ColsC + m] = *alpha * sum + *beta * C[n * ColsC + m];
			}
		}
		A += SizeA;
		B += SizeB;
		C += SizeC;
	}
}

void simpleGEMM(
	bool transA,		// is the input matrix transposed
	bool transB,		// is the weight matrix transposed
	bool transC,		// is the output matrix transposed
	int inHeight,		// the height of output matrix after the operation
	int outWidth,		// the width of output matrix after the operation	
	int shared,			// the leftover dimension
	int batchCount,		// the number of sequantial matrices to process
	float* X,			// the X in Y = X * W
	float* W,			// the W in Y = X * W
	float* Y			// the Y in Y = X * W
)
{
	const float alpha = 1.0f;
	const float beta = 0.0f;
	// assert total size is the same as user input
	if (transC)
	{
		cpuSgemmStridedBatched(
			transB, transA,
			inHeight, outWidth, shared,
			&alpha,
			X, shared, shared * inHeight,
			W, outWidth, shared * outWidth,
			&beta,
			Y, inHeight, inHeight * outWidth,
			batchCount
		);
	}
	else
	{
		cpuSgemmStridedBatched(
			transB, transA,
			outWidth, inHeight, shared,
			&alpha,
			W, outWidth, shared * outWidth,
			X, shared, inHeight * shared,
			&beta,
			Y, outWidth, inHeight * outWidth,
			batchCount
		);
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