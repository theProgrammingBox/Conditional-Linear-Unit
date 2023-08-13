﻿#pragma once

#include <iostream>
#include <vector>
#include <chrono>
#include <assert.h>

#include <cublas_v2.h>
#include <curand.h>
#include <cuda_runtime.h>

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

__global__ void CurandNormalizef32(float* output, uint32_t size, float min, float range)
{
	uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
	if (index < size)
		output[index] = float(*(uint32_t*)(output + index) * range + min);
}

void CurandGenerateUniformf32(curandGenerator_t generator, float* output, uint32_t size, float min = -1.0f, float max = 1.0f)
{
	curandGenerate(generator, (uint32_t*)output, size);
	CurandNormalizef32 << <std::ceil(0.0009765625f * size), 1024 >> > (output, size, min, (max - min) * 2.3283064365387e-10f);
}

__global__ void GpuReluf32(float* input, float* output, uint32_t size)
{
	uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
	if (index < size && *(uint32_t*)(input + index) >> 31)
	{
		output[index] = 0;
	}
}

void CLUForward(float* input, float* output, uint32_t size)
{
	cudaMemcpy(output, input, size << 2, cudaMemcpyDeviceToDevice);
	GpuReluf32 << <std::ceil(0.0009765625f * size), 1024 >> > (input, output, size);
}