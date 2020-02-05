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