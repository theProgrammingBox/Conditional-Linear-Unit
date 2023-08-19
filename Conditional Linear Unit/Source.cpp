#include "NeuralNetwork.cuh"

// rename tensors to tensors

int main()
{
	float learningrate = 0.01f;
	size_t batches = 16;
	size_t inputWidth = 16;
	size_t outputWidth = 8;

	float* hostInputTensor, * hostOutputTensor;
	float* hostOutputGradientTensor, * hostInputGradientTensor;

	NeuralNetwork neuralNetwork
	(
		&learningrate, &batches,
		hostInputTensor, hostOutputTensor,
		hostOutputGradientTensor, hostInputGradientTensor
	);
	neuralNetwork.AddLayer(16, 1, outputWidth, 1);
	neuralNetwork.Initialize(inputWidth, outputWidth);

	printf("CLU\n");

	return 0;
}