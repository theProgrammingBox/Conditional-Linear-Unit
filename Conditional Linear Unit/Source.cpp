#include "NeuralNetwork.cuh"

int main()
{
	/*// malloc gpu arr for 10 elements
	float* deviceArr;
	cudaMalloc(&deviceArr, 10 * sizeof(float));

	// randomize
	GpuRand gpuRand;
	gpuRand.Randomize(deviceArr, 10);

	// print gpu arr
	float* hostArr = (float*)malloc(10 * sizeof(float));
	cudaMemcpy(hostArr, deviceArr, 10 * sizeof(float), cudaMemcpyDeviceToHost);
	PrintTensorf32(10, 1, hostArr, "deviceArr");
	return 0;*/

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
	neuralNetwork.PrintParameters();

	printf("CLU\n");

	return 0;
}