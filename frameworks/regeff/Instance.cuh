#pragma once

#include "TestInstance.cuh"
#include "UtilityFunctions.cuh"
#include "gpualloc_impl.cuh"

enum class RegEffVariants
{
	CudaMalloc,		// - mallocCudaMalloc, freeCudaMalloc
	AtomicMalloc,	// - mallocAtomicMalloc
	AWMalloc,		// - mallocAtomicWrapMalloc
	CMalloc, 		// - mallocCircularMalloc, freeCircularMalloc
	CFMalloc,		// - mallocCircularFusedMalloc, freeCircularFusedMalloc
	CMMalloc,		// - mallocCircularMultiMalloc, freeCircularMultiMalloc
	CFMMalloc		// - mallocCircularFusedMultiMalloc, freeCircularFusedMultiMalloc
};

template <RegEffVariants variant=RegEffVariants::CudaMalloc>
struct MemoryManagerRegEff : public MemoryManagerBase
{
	explicit MemoryManagerRegEff(size_t instantiation_size = 2048ULL*1024ULL*1024ULL) : MemoryManagerBase(instantiation_size) {}
	~MemoryManagerRegEff()
	{
		if(variant != RegEffVariants::CudaMalloc)
		{
			char* m_mallocData{nullptr};
			cudaMemcpyFromSymbol(&m_mallocData, g_heapBase, sizeof(char*));
			cudaFree(m_mallocData);
		}
	}

	virtual void init() override
	{
		if(variant == RegEffVariants::CudaMalloc)
		{
			cudaDeviceSetLimit(cudaLimitMallocHeapSize, size);
			return;
		}

		// ######################################################################
		// Prepare the memory for the heap

		// Set the heapBase
		char* m_mallocData{nullptr};
		cudaMalloc(&m_mallocData, size);
		cudaMemcpyToSymbol(g_heapBase, &m_mallocData, sizeof(char*));

		// Init the heapOffset
		unsigned int _g_heapOffset{0};
		cudaMemcpyToSymbol(g_heapOffset, &_g_heapOffset, sizeof(unsigned int));

		if (variant == RegEffVariants::CMalloc || variant == RegEffVariants::CFMalloc ||
			variant == RegEffVariants::CMMalloc || variant == RegEffVariants::CFMMalloc)
		{
			// Init the heapMultiOffset
			int device{0};
			cudaGetDevice(&device);
			int m_numSM{0};
			cudaDeviceGetAttribute(&m_numSM, cudaDevAttrMultiProcessorCount, device);
			cudaMemcpyToSymbol(g_numSM, &m_numSM, sizeof(unsigned int));

			unsigned int** m_multiOffset{nullptr};
			cudaMalloc(&m_multiOffset, m_numSM * sizeof(unsigned int*));
			cudaMemcpyToSymbol(g_heapMultiOffset, &m_multiOffset, sizeof(unsigned int*));

			// Init the header size
			unsigned int heapSize = size;

			unsigned int headerSize{0};
			if(variant == RegEffVariants::CMalloc || variant == RegEffVariants::CMMalloc)
				headerSize = CIRCULAR_MALLOC_HEADER_SIZE;
			else
				headerSize = sizeof(unsigned int);
			
			unsigned int heapLock{0};
			cudaMemcpyToSymbol(g_heapLock, &heapLock, sizeof(unsigned int));

			// Set the chunk size
			unsigned int numChunks{0};
			unsigned int chunkSize{Utils::alignment<unsigned int>(static_cast<unsigned int>((headerSize + payload) * chunkRatio), ALIGN)};

			// Create hierarchical chunks
			unsigned int minChunkSize = chunkSize;
			float treeMem = static_cast<float>(minChunkSize);
			float heapTotalMem = static_cast<float>(heapSize-2*headerSize);
			float heapMem{0.f};
			int repeats{1};
			if(variant == RegEffVariants::CMalloc || variant == RegEffVariants::CFMalloc)
			{
				heapMem = heapTotalMem;
				repeats = 1;
			}
			else
			{
				heapMem = heapTotalMem/static_cast<float>(m_numSM);
				repeats = m_numSM;
			}

			unsigned int i {1};
			for(; treeMem < heapMem; i++)
			{
				treeMem = static_cast<float>(i + 1) * static_cast<float>(1 << i) * static_cast<float>(minChunkSize);
			}
			chunkSize = (1 << (i - 2)) * minChunkSize;

			// Launch the prepare3 kernel
			if(variant == RegEffVariants::CMalloc || variant == RegEffVariants::CFMalloc)
				numChunks = (1 << (i-1)); 						// Number of nodes of the tree + 1 for the rest
			else
				numChunks = ((1 << (i-1)) - 1) * m_numSM + 1; 	// Number of nodes of the tree times numSM + 1 for the rest
			
			int blockSize = 256;
			int gridSize = numChunks;
			if(variant == RegEffVariants::CMalloc)
			{
				CircularMallocPrepare3 <<<gridSize, blockSize>>> (numChunks, chunkSize);
				cudaDeviceSynchronize();
			}
			else if(variant == RegEffVariants::CFMalloc)
			{
				CircularFusedMallocPrepare3 <<<gridSize, blockSize>>> (numChunks, chunkSize);
				cudaDeviceSynchronize();
			}
			else if(variant == RegEffVariants::CMMalloc)
			{
				CircularMultiMallocPrepare3 <<<gridSize, blockSize>>> (numChunks, chunkSize);
				cudaDeviceSynchronize();
			}
			else if(variant == RegEffVariants::CFMMalloc)
			{
				CircularFusedMultiMallocPrepare3 <<<gridSize, blockSize>>> (numChunks, chunkSize);
				cudaDeviceSynchronize();
			}
		}
	}

	virtual __device__ __forceinline__ void* malloc(size_t size) override
	{
		if(variant == RegEffVariants::CudaMalloc)
		{
			return mallocCudaMalloc(size);
		}
		else if (variant == RegEffVariants::AtomicMalloc)
		{
			return mallocAtomicMalloc(size);
		}
		else if (variant == RegEffVariants::AWMalloc)
		{
			return mallocAtomicWrapMalloc(size);
		}
		else if (variant == RegEffVariants::CMalloc)
		{
			return mallocCircularMalloc(size);
		}
		else if (variant == RegEffVariants::CFMalloc)
		{
			return mallocCircularFusedMalloc(size);
		}
		else if (variant == RegEffVariants::CMMalloc)
		{
			return mallocCircularMultiMalloc(size);
		}
		else if (variant == RegEffVariants::CFMMalloc)
		{
			return mallocCircularFusedMultiMalloc(size);
		}
		else
		{
			printf("Variant not implemented!\n");
			return nullptr;
		}
	}

	virtual __device__ __forceinline__ void free(void* ptr) override
	{
		if(variant == RegEffVariants::CudaMalloc)
		{
			freeCudaMalloc(ptr);
		}
		else if (variant == RegEffVariants::CMalloc)
		{
			freeCircularMalloc(ptr);
		}
		else if (variant == RegEffVariants::CFMalloc)
		{
			freeCircularFusedMalloc(ptr);
		}
		else if (variant == RegEffVariants::CMMalloc)
		{
			freeCircularMultiMalloc(ptr);
		}
		else if (variant == RegEffVariants::CFMMalloc)
		{
			freeCircularFusedMultiMalloc(ptr);
		}
		else
		{
			// variant == RegEffVariants::AtomicMalloc || variant == RegEffVariants::AWMalloc
			// Those variants have no deallocations
		}
	}

	__host__ std::string getDescriptor()
	{
		if(variant == RegEffVariants::CudaMalloc)
		{
			return std::string("RegEff - CudaMalloc");
		}
		else if (variant == RegEffVariants::AtomicMalloc)
		{
			return std::string("RegEff - AtomicMalloc");
		}
		else if (variant == RegEffVariants::AWMalloc)
		{
			return std::string("RegEff - AWMalloc");
		}
		else if (variant == RegEffVariants::CMalloc)
		{
			return std::string("RegEff - CMalloc");
		}
		else if (variant == RegEffVariants::CFMalloc)
		{
			return std::string("RegEff - CFMalloc");
		}
		else if (variant == RegEffVariants::CMMalloc)
		{
			return std::string("RegEff - CMMalloc");
		}
		else if (variant == RegEffVariants::CFMMalloc)
		{
			return std::string("RegEff - CFMMalloc");
		}
	}

	// Found in AppEnvironment.cpp.269
	static constexpr unsigned int payload{4};
	static constexpr double maxFrag{2.0};
	static constexpr double chunkRatio{1.0};
};
