#include "NeuralNetwork.cuh"

/*
TODO:
- Work on backward pass
-- add their grad tensors

- make everything cleaner
-- new drawing using updated names (not code related)
-- use references if possible
-- pass non pointers as not planning to update widths after init
-- seperate ProvideAllocationDetails into connect dimensions and report tensor details
-- comment out debug prints
-- Move gpu rand to a new header file
*/

int main()
{
	float* hostInputTensor, * hostOutputTensor;
	float* hostOutputGradientTensor, * hostInputGradientTensor;
	float learningRate = 0.01f;
	size_t batches = 16;
	size_t inputWidth = 16;
	size_t outputWidth = 8;

	NeuralNetwork neuralNetwork;
	neuralNetwork.AddLayer(new CLU(16, 4, 4, 2, &learningRate));
	neuralNetwork.AddLayer(new CLU(16, 1, outputWidth, 1, &learningRate));
	neuralNetwork.Initialize
	(
		&hostInputTensor, &hostOutputTensor,
		&hostOutputGradientTensor, &hostInputGradientTensor,
		&batches, &inputWidth, &outputWidth
	);
	neuralNetwork.PrintParameters();


	for (size_t i = 0; i < batches * inputWidth; i++)
		hostInputTensor[i] = i / float(batches * inputWidth);

	neuralNetwork.Forward();


	for (size_t i = 0; i < batches * outputWidth; i++)
		hostOutputGradientTensor[i] = i / float(batches * outputWidth);

	neuralNetwork.Backward();


	printf("CLU2\n");

	return 0;
}