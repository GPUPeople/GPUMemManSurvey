#include <iostream>

#include "Instance.cuh"
#include "UtilityFunctions.cuh"

template <typename MemoryManager>
__global__ void d_testFunctions(MemoryManager memory_manager)
{
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	if(tid > 0)
		return;

	int* test_array = reinterpret_cast<int*>(memory_manager.malloc(sizeof(int) * 16));

	printf("%d - %d | Pointer received: %p\n", threadIdx.x, blockIdx.x, test_array);

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
	std::cout << "Simple RegEff Testcase\n";
	
	{
		MemoryManagerRegEff<RegEffVariants::CudaMalloc> memory_manager;

		memory_manager.init();

		d_testFunctions <<<1,1>>>(memory_manager);

		CHECK_ERROR(cudaDeviceSynchronize());

		printf("Testcase CudaMalloc done!\n");
	}

	{
		MemoryManagerRegEff<RegEffVariants::AtomicMalloc> memory_manager;

		memory_manager.init();

		d_testFunctions <<<1,1>>>(memory_manager);

		CHECK_ERROR(cudaDeviceSynchronize());

		printf("Testcase AtomicMalloc done!\n");
	}

	{
		MemoryManagerRegEff<RegEffVariants::AWMalloc> memory_manager;

		memory_manager.init();

		d_testFunctions <<<1,1>>>(memory_manager);

		CHECK_ERROR(cudaDeviceSynchronize());

		printf("Testcase AWMalloc done!\n");
	}

	{
		MemoryManagerRegEff<RegEffVariants::CMalloc> memory_manager;

		memory_manager.init();

		d_testFunctions <<<1,1>>>(memory_manager);

		CHECK_ERROR(cudaDeviceSynchronize());

		printf("Testcase CMalloc done!\n");
	}

	{
		MemoryManagerRegEff<RegEffVariants::CFMalloc> memory_manager;

		memory_manager.init();

		d_testFunctions <<<1,1>>>(memory_manager);

		CHECK_ERROR(cudaDeviceSynchronize());

		printf("Testcase CFMalloc done!\n");
	}

	{
		MemoryManagerRegEff<RegEffVariants::CMMalloc> memory_manager;

		memory_manager.init();

		d_testFunctions <<<1,1>>>(memory_manager);

		CHECK_ERROR(cudaDeviceSynchronize());

		printf("Testcase CMMalloc done!\n");
	}

	{
		MemoryManagerRegEff<RegEffVariants::CFMMalloc> memory_manager;

		memory_manager.init();

		d_testFunctions <<<1,1>>>(memory_manager);

		CHECK_ERROR(cudaDeviceSynchronize());

		printf("Testcase CFMMalloc done!\n");
	}


	return 0;
}

