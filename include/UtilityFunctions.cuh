#pragma once

#include <cuda_runtime_api.h>
#include <device_launch_parameters.h>
#include <cuda_runtime.h>
#include "cub/cub.cuh"
#include <thrust/device_vector.h>
#include <thrust/sort.h>

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

namespace Utils
{
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
}

namespace Helper
{
	template <typename DataType, typename CountType>
	static void cubExclusiveSum(DataType* input_data, CountType num_elements, DataType* output_data = nullptr)
	{
		// Determine temporary device storage requirements
		void     *d_temp_storage = nullptr;
		size_t   temp_storage_bytes = 0;
		CHECK_ERROR(cub::DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes, input_data, output_data ? output_data : input_data, num_elements));
		// Allocate temporary storage
		CHECK_ERROR(cudaMalloc(&d_temp_storage, temp_storage_bytes));
		// Run exclusive prefix sum
		CHECK_ERROR(cub::DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes, input_data, output_data ? output_data : input_data, num_elements));

		CHECK_ERROR(cudaFree(d_temp_storage));
	}

	template <typename DataType, typename CountType>
	static void thrustExclusiveSum(DataType* input_data, CountType num_elements, DataType* output_data = nullptr)
	{
		thrust::device_ptr<DataType> th_data(input_data);
		thrust::device_ptr<DataType> th_output(output_data);
		thrust::exclusive_scan(th_data, th_data + num_elements, output_data ? th_output : th_data);
	}

	template <typename DataType, typename CountType>
	static void thrustSort(DataType* input_data, CountType num_elements)
	{
		thrust::device_ptr<DataType> th_data(input_data);
		thrust::sort(th_data, th_data + num_elements);
	}

	template <typename DataType, typename CountType>
	static void cubSort(DataType* input_data, CountType num_elements, int begin_bit = 0, int end_bit = sizeof(DataType) * 8)
	{
		// Determine temporary device storage requirements
		void     *d_temp_storage = nullptr;
		size_t   temp_storage_bytes = 0;
		CHECK_ERROR(cub::DeviceRadixSort::SortKeys(d_temp_storage, temp_storage_bytes, input_data, input_data, num_elements, begin_bit, end_bit));
		// Allocate temporary storage
		CHECK_ERROR(cudaMalloc(&d_temp_storage, temp_storage_bytes));
		// Run sorting operation
		CHECK_ERROR(cub::DeviceRadixSort::SortKeys(d_temp_storage, temp_storage_bytes, input_data, input_data, num_elements, begin_bit, end_bit));

		CHECK_ERROR(cudaFree(d_temp_storage));
	}

	template <typename DataType, typename ValueType, typename CountType>
	static void cubSortPairs(DataType* input_data, ValueType* input_values, CountType num_elements, int begin_bit = 0, int end_bit = sizeof(DataType) * 8)
	{
		// Determine temporary device storage requirements
		void     *d_temp_storage = nullptr;
		size_t   temp_storage_bytes = 0;
		CHECK_ERROR(cub::DeviceRadixSort::SortPairs(d_temp_storage, temp_storage_bytes, input_data, input_data, input_values, input_values, num_elements, begin_bit, end_bit));
		// Allocate temporary storage
		CHECK_ERROR(cudaMalloc(&d_temp_storage, temp_storage_bytes));
		// Run sorting operation
		CHECK_ERROR(cub::DeviceRadixSort::SortPairs(d_temp_storage, temp_storage_bytes, input_data, input_data, input_values, input_values, num_elements, begin_bit, end_bit));

		CHECK_ERROR(cudaFree(d_temp_storage));
	}
}

