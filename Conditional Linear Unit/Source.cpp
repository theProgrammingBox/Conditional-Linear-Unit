#include "NeuralNetwork.cuh"

int main()
{
	float* hostInputTensor, * hostOutputTensor;
	float* hostOutputGradientTensor, * hostInputGradientTensor;
	float learningrate = 0.01f;
	size_t batches = 16;
	size_t inputWidth = 16;
	size_t outputWidth = 8;

	NeuralNetwork neuralNetwork
	(
		hostInputTensor, hostOutputTensor,
		hostOutputGradientTensor, hostInputGradientTensor,
		&learningrate, &batches
	);
	neuralNetwork.AddLayer(new CLU(inputWidth, 1, outputWidth, 1));
	neuralNetwork.Initialize(&inputWidth, &outputWidth);

	printf("CLU\n");

	return 0;
}