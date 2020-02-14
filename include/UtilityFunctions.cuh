#pragma once

#include <cuda_runtime_api.h>
#include <device_launch_parameters.h>
#include <cuda_runtime.h>

namespace Utils{

static inline void HandleError(cudaError_t err,
	const char* string,
	const char *file,
	int line) {
	if (err != cudaSuccess) {
		printf("%s\n", string);
		printf("%s in \n\n%s at line %d\n", cudaGetErrorString(err),
			file, line);
		exit(EXIT_FAILURE);
	}
}
static inline void HandleError(const char *file,
	int line) {
	auto err = cudaGetLastError();
	if (err != cudaSuccess) {
		printf("%s in %s at line %d\n", cudaGetErrorString(err),
			file, line);
		exit(EXIT_FAILURE);
	}
}
}

#define CHECK_ERROR( err ) (Utils::HandleError( err, "", __FILE__, __LINE__ ))
#define CHECK_ERROR_S( err , string) (Utils::HandleError( err, string, __FILE__, __LINE__ ))

namespace Utils{
// ##############################################################################################################################################
//
void inline start_clock(cudaEvent_t &start, cudaEvent_t &end)
{
	CHECK_ERROR(cudaEventCreate(&start));
	CHECK_ERROR(cudaEventCreate(&end));
	CHECK_ERROR(cudaEventRecord(start, 0));
}

// ##############################################################################################################################################
//
float inline end_clock(cudaEvent_t &start, cudaEvent_t &end)
{
	float time;
	CHECK_ERROR(cudaEventRecord(end, 0));
	CHECK_ERROR(cudaEventSynchronize(end));
	CHECK_ERROR(cudaEventElapsedTime(&time, start, end));
	CHECK_ERROR(cudaEventDestroy(start));
	CHECK_ERROR(cudaEventDestroy(end));

	// Returns ms
	return time;
}
}

// ##############################################################################################################################################
//
template<typename T>
__host__ __device__ __forceinline__ T divup(T a, T b)
{
	return (a + b - 1) / b;
}

// ##############################################################################################################################################
//
template<typename T, typename O>
constexpr __host__ __device__ __forceinline__ T divup(T a, O b)
{
	return (a + b - 1) / b;
}

// ##############################################################################################################################################
//
template<typename T>
constexpr __host__ __device__ __forceinline__ T alignment(const T size, size_t alignment)
{
	return divup<T, size_t>(size, alignment) * alignment;
}