#include "NeuralNetwork.cuh"

/*
TODO:
- redesign and make everything cleaner
-- allow user to pass device tensors
-- allow user to choose to copy device tensors to host
-- use references if possible
-- pass non pointers as not planning to update widths after init

-- comment out debug prints
-- new drawing using updated names (not code related)
-- Move gpu rand to a new header file

- Work on backward pass
-- add their grad tensors
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