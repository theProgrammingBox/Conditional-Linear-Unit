#pragma once
#include <stdio.h>	// printf
#include <stdlib.h>	// malloc
#include <time.h>	// time
#include <stdint.h>	// uint32_t
#include <assert.h>	// assert
#include <vector>

#include <cublas_v2.h>
#include <curand.h>
#include <cuda_runtime.h>

typedef uint32_t u32;
typedef int8_t i8;
typedef float f32;