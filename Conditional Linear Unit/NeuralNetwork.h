#pragma once
#include "CLU.cuh"

struct NeuralNetwork
{
	cublasHandle_t cublasHandle;
	curandGenerator_t curandGenerator;

	int maxInHeight;
	float* input, * outputGrad;
	std::vector<CLU*> layers;

	NeuralNetwork()
	{
		cublasCreate(&cublasHandle);

		curandCreateGenerator(&curandGenerator, CURAND_RNG_PSEUDO_DEFAULT);
		curandSetPseudoRandomGeneratorSeed(curandGenerator, std::chrono::high_resolution_clock::now().time_since_epoch().count());
	}

	~NeuralNetwork()
	{
		cublasDestroy(cublasHandle);
		curandDestroyGenerator(curandGenerator);

		for (int i = 0; i < layers.size(); ++i)
			delete layers[i];
	}

	void AddLayer(int inWidth, int hiddenWidth, int hiddenHeight, int outWidth, int heads)
	{
		layers.push_back
		(
			new CLU
			(
				&cublasHandle, &curandGenerator,
				inWidth, hiddenWidth, hiddenHeight, outWidth, heads
			)
		);
	}

	void Compile()
	{
		size_t free, total;
		cudaMemGetInfo(&free, &total);
		printf("free: %lu\n", free);
		printf("total: %lu\n", total);

		int denominator = layers.front()->GetInputWidth() + layers.back()->GetOutputWidth();
		for (int i = 0; i < layers.size(); ++i)
			denominator += layers[i]->GetSizeCoefficient();
		printf("denominator: %d\n", denominator);

		maxInHeight = free / denominator;
		printf("maxInHeight: %d\n", maxInHeight);

		for (int i = 0; i < layers.size(); ++i)
			layers[i]->Allocate(maxInHeight);

		cudaMalloc(&input, sizeof(float) * layers.front()->GetInputWidth() * maxInHeight);
		cudaMalloc(&outputGrad, sizeof(float) * maxInHeight * layers.back()->GetOutputWidth());

		layers.front()->input = input;
		for (int i = 1; i < layers.size() - 1; ++i)
		{
			layers[i]->input = layers[i - 1]->output;
			layers[i - 1]->outputGrad = layers[i]->inputGrad;
		}
		layers.back()->outputGrad = outputGrad;
	}

	void Forward(int inHeight, float* input)
	{
		assert(inHeight <= maxInHeight);

		// cuda memcpy cpu to gpu
		cudaMemcpy(this->input, input, sizeof(float) * inHeight * layers[0]->inWidth, cudaMemcpyHostToDevice);

		// cuda memcpy gpu to cpu

	}

	void Backward(float learningrate)
	{
	}

	void PrintParameters()
	{
		for (int i = 0; i < layers.size(); ++i)
			layers[i]->PrintParameters();
	}
};