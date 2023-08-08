#include "Header.h"

struct CLU
{
	int * inputHeight, * inputWidth;
	int hiddenHeight, hiddenWidth, outputWidth;
	float* x, * weight, * bias, * product, * y;
	static const float ONEF;
	static const float ZEROF;

	CLU(int* inputHeight, int* inputWidth, int hiddenHeight, int hiddenWidth, int outputWidth, float* x)
	{
		this->inputHeight = inputHeight;
		this->inputWidth = inputWidth;
		this->hiddenHeight = hiddenHeight;
		this->hiddenWidth = hiddenWidth;
		this->outputWidth = outputWidth;
		this->x = x;
		weight = new float[hiddenHeight * hiddenWidth];
		bias = new float[hiddenHeight];
		product = new float[hiddenHeight];
		y = new float[outputWidth];
	}
};

const float CLU::ONEF = 1.0f;
const float CLU::ZEROF = 0.0f;