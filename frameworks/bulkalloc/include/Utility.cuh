#pragma once


namespace BUtils
{
	template <unsigned int ALIGNMENT>
	__forceinline__ __device__ bool isAlignedToSize(void* base, void* ptr)
	{
		if ((reinterpret_cast<unsigned long long>(ptr) - reinterpret_cast<unsigned long long>(base)) & (ALIGNMENT - 1))
			return false;
		else
			return true;
	}
}
