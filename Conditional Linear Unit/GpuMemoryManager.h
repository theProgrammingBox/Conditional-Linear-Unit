#pragma once
#include "Header.cuh"

struct GpuMemoryManager
{
	struct MemFrag
	{
		size_t size;
		float* address;
	};

	struct TensorData
	{
		float** address;
		size_t size;
	};

	std::vector<MemFrag*> MemFrags;
	std::vector<TensorData*> tensorPtrs;

	GpuMemoryManager()
	{
		size_t freeMem, totalMem;
		cudaMemGetInfo(&freeMem, &totalMem);

		MemFrag* frag;
		size_t low, high, guess;
		cudaError_t err;
		do
		{
			frag = new MemFrag;
			low = 1, high = freeMem;
			do
			{
				guess = (low + high) >> 1;
				err = cudaMalloc((void**)&frag->address, guess);
				err == cudaSuccess ? low = guess + 1 : high = guess - 1;
				cudaFree(frag->address);
			} while (low <= high);
			low--;

			if (low > 0)
			{
				frag->size = low;
				cudaMalloc((void**)&frag->address, low);
				freeMem -= low;
				MemFrags.emplace_back(frag);
			}
		} while (low > 0);
		delete frag;
	}

	~GpuMemoryManager()
	{
		for (MemFrag* frag : MemFrags)
		{
			cudaFree(frag->address);
			delete frag;
		}
	}

	void PrintGpuMem() const
	{
		for (MemFrag* frag : MemFrags)
			printf("Allocated %zu bytes at %p\n", frag->size, frag->address);
		printf("\n");

		for (TensorData* tensorData : tensorPtrs)
			printf("Tensor at %p with size %zu\n", tensorData->address, tensorData->size);
		printf("\n");
	}

	void Manage(float** tensorPtr, size_t size)
	{
		TensorData* tensorData = new TensorData;
		tensorData->address = tensorPtr;
		tensorData->size = size;
		tensorPtrs.emplace_back(tensorData);
	}
};