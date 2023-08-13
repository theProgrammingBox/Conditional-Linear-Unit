#pragma once
#include "CLU.cuh"

struct NeuralNetwork
{
	cublasHandle_t cublasHandle;
	curandGenerator_t curandGenerator;

	int inWidth, outWidth;
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

	void ExpectInWidth(int inWidth)
	{
		this->inWidth = inWidth;
	}

	void ExpectOutWidth(int outWidth)
	{
		this->outWidth = outWidth;
	}

	void AddLayer(int hiddenWidth, int hiddenHeight, int outWidth, int heads)
	{
		layers.push_back
		(
			new CLU
			(
				&cublasHandle, &curandGenerator,
				hiddenWidth, hiddenHeight, outWidth, heads
			)
		);
	}

	void Compile()
	{
		layers.front()->inWidth = inWidth;
		for (int i = 1; i < layers.size() - 1; ++i)
			layers[i]->inWidth = layers[i - 1]->GetOutputWidth();

		size_t free, total;
		cudaMemGetInfo(&free, &total);
		printf("free: %lu\n", free);
		printf("total: %lu\n", total);

		int denominator = inWidth + layers.back()->GetOutputWidth();
		for (int i = 0; i < layers.size(); ++i)
			denominator += layers[i]->GetSizeCoefficient();
		printf("denominator: %d\n", denominator);

		maxInHeight = free / denominator;
		printf("maxInHeight: %d\n\n", maxInHeight);

		for (int i = 0; i < layers.size(); ++i)
			layers[i]->Allocate(maxInHeight);

		cudaMalloc(&input, layers.front()->GetInputWidth() * maxInHeight * sizeof(float));
		cudaMalloc(&outputGrad, layers.back()->GetOutputWidth() * maxInHeight * sizeof(float));

		cudaMemGetInfo(&free, &total);
		printf("free: %lu\n", free);
		printf("total: %lu\n", total);

		layers.front()->input = input;
		for (int i = 1; i < layers.size() - 1; ++i)
		{
			layers[i]->input = layers[i - 1]->output;
			layers[i - 1]->outputGrad = layers[i]->inputGrad;
		}
		layers.back()->outputGrad = outputGrad;

		assert(layers.back()->GetOutputWidth() == outWidth);
	}

	float* Forward(int inHeight, float* input)
	{
		assert(inHeight <= maxInHeight);

		// cuda memcpy cpu to gpu
		cudaMemcpy(this->input, input, inWidth * inHeight * sizeof(float), cudaMemcpyHostToDevice);

		// cuda memcpy gpu to cpu
		float* output = new float[inHeight * layers.back()->outWidth];
		cudaMemcpy(output, layers.back()->output, layers.back()->outWidth * inHeight * sizeof(float), cudaMemcpyDeviceToHost);
		
		return output;
	}

	float* Backward(float learningrate, int inHeight, float* outputGrad)
	{
		assert(inHeight <= maxInHeight);

		// cuda memcpy cpu to gpu
		cudaMemcpy(this->outputGrad, outputGrad, layers.back()->outWidth * inHeight * sizeof(float), cudaMemcpyHostToDevice);

		// cuda memcpy gpu to cpu
		float* inputGrad = new float[inHeight * layers.front()->inWidth];
		cudaMemcpy(inputGrad, layers.front()->inputGrad, layers.front()->inWidth * inHeight * sizeof(float), cudaMemcpyDeviceToHost);

		return inputGrad;
	}

	void PrintParameters()
	{
		for (int i = 0; i < layers.size(); ++i)
			layers[i]->PrintParameters();
	}
};