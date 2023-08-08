#include "Header.h"

// clean up variables, make sure they are initialized after implementation

/*
struct CLU
{
	int * inputHeight, * inputWidth, * inputSize;
	int hiddenHeight, hiddenWidth;
	int outputWidth;
	int trueHiddenWidth, trueOutputWidth;
	int productSize;
	float invInputWidth;
	float* x, * weight, * bias, * product, * y;
	static constexpr float zero = 0.0f;

	CLU(int* inputHeight, int* inputWidth, int hiddenHeight, int hiddenWidth, int outputWidth, float* x)
	{
		this->inputHeight = inputHeight;
		this->inputWidth = inputWidth;
		this->hiddenHeight = hiddenHeight;
		this->hiddenWidth = hiddenWidth;
		this->outputWidth = outputWidth;
		invInputWidth = 1.0f / *inputWidth;
		this->x = x;
		trueHiddenWidth = hiddenWidth * (hiddenHeight + outputWidth);
		weight = new float[*inputWidth * trueHiddenWidth];
		bias = new float[hiddenHeight];
		product = new float[hiddenHeight];
		y = new float[outputWidth];
	}

	~CLU()
	{
		delete[] weight;
		delete[] bias;
		delete[] product;
		delete[] y;
	}

	void forward(float* alpha, float* beta)
	{
		cpuSgemmStridedBatched
		(
			false, false,
			*inputHeight, trueHiddenWidth, *inputWidth,
			&invInputWidth,
			x, *inputHeight, 0,
			weight, *inputWidth, 0,
			&zero,
			product, *inputHeight, 0,
			1
		);
	}
};
*/

int main()
{
	/*
	char a = 'a', b = 'b', c = 'c';

	// 6 possible combinations of a, b, c
	std::vector<std::vector<char>> primary = {
		{a, b, c},
		{a, c, b},
		{b, a, c},
		{b, c, a},
		{c, a, b},
		{c, b, a}
	};

	std::vector<std::vector<char>> result;

	// For each primary combination
	for (const auto& p : primary) {
		// Consider all possible combinations of the independent variables
		for (int i = 0; i < 2; ++i) {
			for (int j = 0; j < 2; ++j) {
				for (int k = 0; k < 2; ++k) {
					result.push_back({
						p[0], p[1], p[2], // the primary combination
						(i == 0 ? a : b), // a or b
						(j == 0 ? b : c), // b or c
						(k == 0 ? c : a)  // c or a
						});
				}
			}
		}
	}

	// Print the resulting combinations
	for (const auto& comb : result) {
		for (char ch : comb) {
			std::cout << ch << ' ';
		}
		std::cout << '\n';
	}

	return 0;
	*/


	const float one = 1.0f, zero = 0.0f;

	int ins = 3, shared = 2, outs = 3;

	std::vector<float> X(ins * shared, 0);
	std::vector<float> W(shared * outs, 0);
	std::vector<float> Y(ins * outs, 0);
	std::vector<float> T(ins * outs, 0);

	for (int i = 0; i < ins * shared; ++i)
		X[i] = i;
	for (int i = 0; i < shared * outs; ++i)
		W[i] = i;

	PrintMatrixf32(X.data(), ins, shared, "X");

	PrintMatrixf32(W.data(), shared, outs, "W");

	cpuSgemmStridedBatched
	(
		false, false,
		outs, ins, shared,
		&one,
		W.data(), outs, 0,
		X.data(), shared, 0,
		&zero,
		Y.data(), outs, 0,
		1
	);
	PrintMatrixf32(Y.data(), ins, outs, "Y");

	cpuSgemmStridedBatched
	(
		true, false,
		shared, ins, outs,
		&one,
		W.data(), shared, 0,	// can be either shared or outs
		X.data(), shared, 0,	// can be either shared or ins
		&zero,
		T.data(), outs, 0,		// can be either outs or ins
		1
	);
	PrintMatrixf32(T.data(), ins, outs, "T");

	cpuSgemmStridedBatched
	(
		true, false,
		shared, ins, outs,
		&one,
		X.data(), shared, 0,
		W.data(), shared, 0,
		&zero,
		T.data(), outs, 0,
		1
	);
	PrintMatrixf32(T.data(), ins, outs, "T");

	cpuSgemmStridedBatched
	(
		false, true,
		shared, ins, outs,
		&one,
		W.data(), shared, 0,
		X.data(), shared, 0,
		&zero,
		T.data(), outs, 0,
		1
	);
	PrintMatrixf32(T.data(), ins, outs, "T");

	cpuSgemmStridedBatched
	(
		false, true,
		shared, ins, outs,
		&one,
		X.data(), shared, 0,
		W.data(), shared, 0,
		&zero,
		T.data(), outs, 0,
		1
	);
	PrintMatrixf32(T.data(), ins, outs, "T");

	cpuSgemmStridedBatched
	(
		true, true,
		ins, outs, shared,
		&one,
		X.data(), shared, 0,
		W.data(), outs, 0,
		&zero,
		T.data(), ins, 0,
		1
	);
	PrintMatrixf32(T.data(), ins, outs, "T");

	return 0;
}