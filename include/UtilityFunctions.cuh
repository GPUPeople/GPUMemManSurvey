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

	// ##############################################################################################################################################
	//
	static constexpr int cntlz(unsigned int x)
	{
		if (x == 0) return 32;
		int n = 0;
		if (x <= 0x0000FFFF) { n = n + 16; x = x << 16; }
		if (x <= 0x00FFFFFF) { n = n + 8; x = x << 8; }
		if (x <= 0x0FFFFFFF) { n = n + 4; x = x << 4; }
		if (x <= 0x3FFFFFFF) { n = n + 2; x = x << 2; }
		if (x <= 0x7FFFFFFF) { n = n + 1; x = x << 1; }
		return n;
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

	template<unsigned int X, int Completed = 0>
	struct static_clz
	{
		static const int value = (X & 0x80000000) ? Completed : static_clz< (X << 1), Completed + 1 >::value;
	};
	template<unsigned int X>
	struct static_clz<X, 32>
	{
		static const int value = 32;
	};

	struct AllocationHelper
	{
		static __host__ int __host_clz(unsigned int x)
		{
			int n = 32;
			unsigned y;

			y = x >>16; if (y != 0) { n = n -16; x = y; }
			y = x >> 8; if (y != 0) { n = n - 8; x = y; }
			y = x >> 4; if (y != 0) { n = n - 4; x = y; }
			y = x >> 2; if (y != 0) { n = n - 2; x = y; }
			y = x >> 1; if (y != 0) return n - 2;
			return n - x;
		}

		template <typename T>
		static __device__ __host__ __forceinline__ int getNextPow2(T n)
		{
			return 1 << (getNextPow2Pow(n));
		}

		template <typename T>
		static __device__ __host__ __forceinline__ int getNextPow2Pow(T n)
		{
			#ifdef __CUDA_ARCH__
			if ((n & (n - 1)) == 0)
				return 32 - __clz(n) - 1;
			else
				return 32 - __clz(n);
			#else
			if ((n & (n - 1)) == 0)
				return 32 - __host_clz(n) - 1;
			else
				return 32 - __host_clz(n);
			#endif
		}

		template <typename T, int minPageSize>
		static __device__ __host__ __forceinline__ int getPageSize(T n)
		{
			return max(minPageSize, getNextPow2(n));
		}

		template <typename T, T n>
		static constexpr int static_getNextPow2Pow()
		{
			if ((n & (n - 1)) == 0)
				return 32 - static_clz<n>::value - 1;
			else
				return 32 - static_clz<n>::value;
		}

		template <typename T, T n>
		static constexpr size_t static_getNextPow2()
		{
			return 1 << (static_getNextPow2Pow(n));
		}
	};
}

static constexpr char PBSTR[] = "##############################################################################################################";
static constexpr int PBWIDTH = 99;

static inline void printProgressBar(const double percentage)
{
	auto val = static_cast<int>(percentage * 100);
	auto lpad = static_cast<int>(percentage * PBWIDTH);
	auto rpad = PBWIDTH - lpad;
#ifdef WIN32
	printf("\r%3d%% [%.*s%*s]", val, lpad, PBSTR, rpad, "");
#else
	printf("\r\033[0;35m%3d%% [%.*s%*s]", val, lpad, PBSTR, rpad, "");
#endif
	fflush(stdout);
}
static inline void printProgressBarEnd()
{
#ifdef WIN32
	printf("\n");
#else
	printf("\033[0m\n");
#endif
	fflush(stdout);
}