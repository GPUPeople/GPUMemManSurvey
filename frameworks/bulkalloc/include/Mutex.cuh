#pragma once

struct Mutex
{
	enum class MutexStates : int
	{
		LOCK,
		UNLOCK
	};

	__device__ __forceinline__ void lock()
	{
		#if (__CUDA_ARCH__ >= 700)
		while(atomicCAS(&state, static_cast<int>(MutexStates::LOCK), static_cast<int>(MutexStates::UNLOCK)) != static_cast<int>(MutexStates::LOCK))
		{
			__nanosleep(10);
		}
		#else
		printf("Cannot do locking prior to CC 7.0\n");
		__trap();
		#endif
	}

	__device__ __forceinline__ void unlock()
	{
		#if (__CUDA_ARCH__ >= 700)
		atomicExch(&state, static_cast<int>(MutexStates::UNLOCK));
		#else
		printf("Cannot do locking prior to CC 7.0\n");
		__trap();
		#endif
	}

	int state{static_cast<int>(MutexStates::UNLOCK)};
};