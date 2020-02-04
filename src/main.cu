#include <iostream>

#include "Utility.cuh"

// ########################
#ifdef TEST_CUDA
#include "cuda/Instance.cuh"
#elif TEST_HALLOC
#include "halloc/Instance.cuh"
#elif TEST_SCATTERALLOC
#include "scatteralloc/Instance.cuh"
#endif

template <typename MemoryManagerType>
__global__ void d_testFunctions(MemoryManagerType memory_manager)
{
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	if(tid > 0)
		return;

	auto test_array = reinterpret_cast<int*>(memory_manager.malloc(sizeof(int) * 16));

	for(int i = 0; i < 16; ++i)
	{
		test_array[i] = i;
	}

	memory_manager.free(test_array);
	return;
}

int main(int argc, char* argv[])
{
#ifdef TEST_CUDA
	std::cout << "--- CUDA ---\n";
	MemoryManagerCUDA memory_manager;
#elif TEST_HALLOC
	std::cout << "--- Halloc ---\n";
	MemoryManagerHalloc memory_manager;
#elif TEST_SCATTERALLOC
	std::cout << "--- ScatterAlloc ---\n";
	MemoryManagerScatterAlloc memory_manager;
#endif

	memory_manager.init();

	d_testFunctions <<<1,1>>>(memory_manager);

	HANDLE_ERROR(cudaDeviceSynchronize());

	printf("Testcase done!\n");

	return 0;
}