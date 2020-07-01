#pragma once

static constexpr int SLEEP_TIME{10};
static constexpr int WARP_SIZE{32};
using memory_t = int8_t;

namespace BUtils
{
	// ##############################################################################################################################################
	//
	template<unsigned int X, int Completed = 0>
	struct static_clz
	{
		static const int value = (X & 0x80000000) ? Completed : static_clz< (X << 1), Completed + 1 >::value;
	};

	// ##############################################################################################################################################
	//
	template<unsigned int X>
	struct static_clz<X, 32>
	{
		static const int value = 32;
	};

	static __device__ __forceinline__ int getNextPow2Pow(unsigned int n)
	{
		if ((n & (n - 1)) == 0)
			return 32 - __clz(n) - 1;
		else
			return 32 - __clz(n);
	}

	static __device__ __forceinline__ int getNextPow2Pow(size_t n)
	{
		if ((n & (n - 1)) == 0)
			return 64 - __clzll(n) - 1;
		else
			return 64 - __clzll(n);
	}

	template <typename T>
	static __device__ __forceinline__ T getNextPow2(T n)
	{
		return 1 << (getNextPow2Pow(n));
	}

	template <unsigned int n>
	static constexpr unsigned int static_getNextPow2Pow()
	{
		if ((n & (n - 1)) == 0)
			return 32 - static_clz<n>::value - 1;
		else
			return 32 - static_clz<n>::value;
	}

	// ##############################################################################################################################################
	//
	template<unsigned long long X, unsigned long long Completed = 0ULL>
	struct static_clz_l
	{
		static const unsigned long long value = (X & 0x8000000000000000) ? Completed : static_clz_l< (X << 1ULL), Completed + 1ULL >::value;
	};

	// ##############################################################################################################################################
	//
	template<unsigned long long X>
	struct static_clz_l<X, 64ULL>
	{
		static const unsigned long long value = 64ULL;
	};

	template <unsigned long long n>
	static constexpr unsigned int static_getNextPow2Pow_l()
	{
		if ((n & (n - 1)) == 0)
			return 64 - static_clz_l<n>::value - 1;
		else
			return 64 - static_clz_l<n>::value;
	}

	template <unsigned int ALIGNMENT>
	__forceinline__ __device__ bool isAlignedToSize(void* base, void* ptr)
	{
		if ((reinterpret_cast<unsigned long long>(ptr) - reinterpret_cast<unsigned long long>(base)) & (ALIGNMENT - 1))
			return false;
		else
			return true;
	}

	static constexpr __forceinline__ __device__ unsigned long long create2Complement(unsigned long long value)
	{
		return ~(value) + 1ULL;
	}

	__forceinline__ __device__ unsigned int ldg_cg(const unsigned int* src)
	{
		unsigned int dest{ 0 };
	#ifdef __CUDA_ARCH__
		asm volatile("ld.global.cg.u32 %0, [%1];" : "=r"(dest) : "l"(src));
	#endif
		return dest;
	}

	__forceinline__ __device__ int ldg_cg(const int* src)
		{
			int dest{ 0 };
	#ifdef __CUDA_ARCH__
			asm volatile("ld.global.cg.s32 %0, [%1];" : "=r"(dest) : "l"(src));
	#endif
			return dest;
		}

	__forceinline__ __device__ unsigned long long ldg_cg(const unsigned long long* src)
	{
		unsigned long long dest{0};
	#ifdef __CUDA_ARCH__
		asm volatile("ld.global.cg.u64 %0, [%1];" : "=l"(dest) : "l"(src));
	#endif
		return dest;
	}

	__forceinline__ __device__ void store(volatile uint4* dest, const uint4& src)
	{
	#ifdef __CUDA_ARCH__
		asm("st.volatile.v4.u32 [%0], {%1, %2, %3, %4};"
			:
		: "l"(dest), "r"(src.x), "r"(src.y), "r"(src.z), "r"(src.w));
	#endif
	}

	__forceinline__ __device__ void store(volatile uint2* dest, const uint2& src)
	{
	#ifdef __CUDA_ARCH__
		asm("st.volatile.v2.u32 [%0], {%1, %2};"
			:
		: "l"(dest), "r"(src.x), "r"(src.y));
	#endif
	}

	static __forceinline__ __device__ int lane_id()
	{
		return threadIdx.x & (WARP_SIZE - 1);
	}

	__forceinline__ __device__ void sleep(unsigned int factor = 1)
	{
	#ifdef __CUDA_ARCH__
	#if (__CUDA_ARCH__ >= 700)
		//__nanosleep(SLEEP_TIME);
		__nanosleep(SLEEP_TIME * factor);
	#else
		__threadfence();
	#endif
	#endif
	}

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

#define CHECK_ERROR( err ) (BUtils::HandleError( err, "", __FILE__, __LINE__ ))
#define CHECK_ERROR_S( err , string) (BUtils::HandleError( err, string, __FILE__, __LINE__ ))
