#include "Header.h"

/*
IMPORTANT LESSONS:
-- In cpuSgemmStridedBatched,
---- there are 3 major dimensions
---- the orientation of the output is deterined by the 3rd and 4th arguments, the 5th is the leftover dimension
---- the first 2 arguments are also important in determining the 8th and 11th arguments
---- the 8th, 11th, and 15th arguments are the strides of the 3 major dimensions
------ this means, how many increments of the pointer to the data are needed to increment the 1st dimension for each matrix
---- the 9th, 12th, and 16th is the number of increments needed to increment the 2nd dimension for each matrix
---- the general rule for basic matmul is:
------ cpuSgemmStridedBatched
------ (
------		bool1, bool2,
------		d1, d2, d3,
------		&one,
------		W, bool1 ? d2 : d3, d2 * d3,
------		X, bool2 ? d1 : d2, d1 * d2,
------		&zero,
------		Y, d1, d1 * d2,
------		n
------ )
*/

int main()
{
	float one = 1.0f, zero = 0.0f;
	int ins = 4, shared = 2, outs = 3;

	std::vector<float> X(ins * shared, 0);
	std::vector<float> W(shared * outs, 0);
	std::vector<float> Y(ins * outs, 0);

	for (int i = 0; i < ins * shared; ++i)
		X[i] = i;
	for (int i = 0; i < shared * outs; ++i)
		W[i] = i;

	PrintMatrixf32(X.data(), ins, shared, "X");
	PrintMatrixf32(W.data(), shared, outs, "W");

	simpleGEMM(
		false, false, false,
		ins, outs, shared, 1,
		X.data(), W.data(), Y.data()
	);

	PrintMatrixf32(Y.data(), ins, outs, "Y");

	return 0;
}