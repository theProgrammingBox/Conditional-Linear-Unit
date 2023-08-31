#include "NeuralNetwork.cuh"

/*
TODO:
- Work on forward pass
-- Work on gpu rand kernel range
-- add custom add and activation kernels
-- Work on forward scalars
- Move gpu rand to a new header file
- Work on backward pass
-- add their dev tensors
- rework data passing
-- use references if possible
*/

int main()
{
	float* hostInputTensor, * hostOutputTensor;
	float* hostOutputGradientTensor, * hostInputGradientTensor;
	float learningrate = 0.01f;
	size_t batches = 16;
	size_t inputWidth = 16;
	size_t outputWidth = 8;

	NeuralNetwork neuralNetwork;
	neuralNetwork.AddLayer(new CLU(16, 4, 4, 2, &learningrate));
	neuralNetwork.AddLayer(new CLU(16, 1, outputWidth, 1, &learningrate));
	neuralNetwork.Initialize
	(
		&hostInputTensor, &hostOutputTensor,
		&hostOutputGradientTensor, &hostInputGradientTensor,
		&batches, &inputWidth, &outputWidth
	);
	neuralNetwork.PrintParameters();

	//memset(hostInputTensor, 0, sizeof(float) * batches * inputWidth);
	for (size_t i = 0; i < batches * inputWidth; i++)
		hostInputTensor[i] = i / float(batches * inputWidth);
	memset(hostOutputGradientTensor, 0, sizeof(float) * batches * outputWidth);

	neuralNetwork.Forward();
	//neuralNetwork.Backward();

	printf("CLU2\n");

	return 0;
}