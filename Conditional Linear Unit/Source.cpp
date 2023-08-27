#include "NeuralNetwork.cuh"

/*
TODO:
- Work on forward pass
- Work on forward scalars
- Move gpu rand to a new header file
- Work on gpu rand kernel range
- Work on backward pass
-- add their dev tensors
*/

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
	neuralNetwork.AddLayer(new CLU(16, 4, 4, 2));
	neuralNetwork.AddLayer(new CLU(16, 1, outputWidth, 1));
	neuralNetwork.Initialize(&inputWidth, &outputWidth);
	neuralNetwork.PrintParameters();

	printf("CLU\n");

	return 0;
}