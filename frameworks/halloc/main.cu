#include <iostream>

#include "Instance.cuh"

template <typename MemoryManager>
__global__ void d_testFunctions(MemoryManager memory_manager)
{
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	if(tid > 1)
		return;

	int* test_array{nullptr};
	if(tid == 0)
		test_array = reinterpret_cast<int*>(memory_manager.malloc(sizeof(int) * 16));
	else
		test_array = reinterpret_cast<int*>(memory_manager.malloc(sizeof(int) * 32));

	for(int i = 0; i < 16; ++i)
	{
		test_array[i] = i;
	}

	memory_manager.free(test_array);

	printf("It worked!\n");

	return;
}

int main(int argc, char* argv[])
{
	std::cout << "Simple Halloc Testcase\n";

	MemoryManagerHalloc memory_manager;

	d_testFunctions <<<1,32>>>(memory_manager);

	cudaDeviceSynchronize();

	printf("Testcase done!\n");

	return 0;
}