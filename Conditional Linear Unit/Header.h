﻿#pragma once
#include <stdio.h>	// printf
#include <stdlib.h>	// malloc
#include <time.h>	// time
#include <stdint.h>	// uint32_t
#include <math.h>	// ceil
#include <vector>

#include <cublas_v2.h>
#include <cuda_runtime.h>

void FailIf(bool condition, const char* message)
{
	if (condition)
	{
		fprintf(stderr, "%s", message);
		exit(0);
	}
}

void PrintTensorf32(size_t width, size_t height, float* arr, const char* label = "Tensor", size_t majorStride = 0, size_t tensorSize = 0, size_t batchCount = 1)
{
	if (majorStride == 0)
		majorStride = width;
	printf("%s:\n", label);
	for (size_t b = batchCount; b--;)
	{
		for (size_t i = 0; i < height; i++)
		{
			for (size_t j = 0; j < width; j++)
				printf("%6.3f ", arr[i * majorStride + j]);
			printf("\n");
		}
		printf("\n");
		arr += tensorSize;
	}
}

__global__ void gpuRandFunc(float* arr, uint32_t size, uint32_t seed1, uint32_t seed2)
{
	uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < size)
	{
		uint32_t index = idx;

		index ^= seed1;
		index *= 0xBAC57D37;
		index ^= index >> 16;
		index ^= seed2;
		index *= 0x24F66AC9;
		index ^= index >> 16;

		arr[idx] = index * 0.00000000023283064365386962890625f;
	}
}

struct GpuRand
{
	uint32_t seed1, seed2;

	void Lehmer32(uint32_t& x)
	{
		x *= 0xBAC57D37;
		x ^= x >> 16;
		x *= 0x24F66AC9;
		x ^= x >> 16;
	}

	GpuRand()
	{
		uint32_t seed1 = time(NULL) ^ 0xE621B963;
		Lehmer32(seed1);
		Lehmer32(seed1);
		uint32_t seed2 = seed1 ^ 0x6053653F ^ (time(NULL) >> 32);
		Lehmer32(seed2);

		printf("Seed1: %u\n", seed1);
		printf("Seed2: %u\n\n", seed2);
	}

	void Randomize(float* arr, uint32_t size)
	{
		Lehmer32(seed1);
		Lehmer32(seed2);
		gpuRandFunc << <ceil(0.0009765625f * size), 1024 >> > (arr, size, seed1, seed2);
	}
};