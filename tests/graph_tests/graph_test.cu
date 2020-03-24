#include <iostream>
#include <string>
#include <fstream>
#include <map>
#include <fstream>
#include <iomanip>
#include <ctime>
#include <sstream>

#include "CSR.h"
#include "dCSR.h"
#include "COO.h"
#include "device/dynamicGraph_impl.cuh"
#include "Definitions.h"

// Json Reader
#include "json.h"
using json = nlohmann::json;

// ########################
#ifdef TEST_CUDA
#include "cuda/Instance.cuh"
#elif TEST_HALLOC
#include "halloc/Instance.cuh"
#elif TEST_SCATTERALLOC
#include "scatteralloc/Instance.cuh"
#elif TEST_OUROBOROS
#include "ouroboros/Instance.cuh"
#elif TEST_FDG
#include "fdg/Instance.cuh"
#elif TEST_REGEFF
#include "regeff/Instance.cuh"
#endif

using DataType = float;
template <typename T>
std::string typeext();

template <>
std::string typeext<float>()
{
	return std::string("");
}

template <>
std::string typeext<double>()
{
	return std::string("d_");
}

int main(int argc, char* argv[])
{
    if (argc == 1)
	{
		std::cout << "Require config file as first argument" << std::endl;
		return -1;
    }

    printf("%sDynamic Graph Example\n%s", CLHighlight::break_line_blue_s, CLHighlight::break_line_blue_e);

    // Parse config
	std::ifstream json_input(argv[1]);
	json config;
    json_input >> config;
    
    const auto device{config.find("device").value().get<int>()};
	cudaSetDevice(device);
	cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device);
    
    auto graphs = *config.find("graphs");
    for(auto const& elem : graphs)
	{
        std::string filename = elem.find("filename").value().get<std::string>();
        CSR<DataType> csr_mat;
        std::string csr_name = filename + typeext<DataType>() + ".csr";
        try
		{
			std::cout << "trying to load csr file \"" << csr_name << "\"\n";
			csr_mat = loadCSR<DataType>(csr_name.c_str());
			std::cout << "succesfully loaded: \"" << csr_name << "\"\n";
		}
		catch (std::exception& ex)
		{
			std::cout << "could not load csr file:\n\t" << ex.what() << "\n";
			try
			{
				std::cout << "trying to load mtx file \"" << filename << "\"\n";
				auto coo_mat = loadMTX<DataType>(filename.c_str());
				convert(csr_mat, coo_mat);
				std::cout << "succesfully loaded and converted: \"" << csr_name << "\"\n";
			}
			catch (std::exception& ex)
			{
				std::cout << ex.what() << std::endl;
				return -1;
			}
			try
			{
				std::cout << "write csr file for future use\n";
				storeCSR(csr_mat, csr_name.c_str());
			}
			catch (std::exception& ex)
			{
				std::cout << ex.what() << std::endl;
			}
        }
        printf("Using: %s with %llu vertices and %llu edges\n", filename, csr_mat.rows, csr_mat.nnz);
        auto max_adjacency_length = 0U;
        auto min_adjacency_length = 0xFFFFFFFFU;
        for(auto i = 0U; i < csr_mat.rows; ++i)
        {
            auto neighbours = csr_mat.row_offsets[i + 1] - csr_mat.row_offsets[i];
            max_adjacency_length = std::max(max_adjacency_length, neighbours);
            min_adjacency_length = std::min(min_adjacency_length, neighbours);
        }
        printf("Smallest Adjacency: %u | Largest Adjacency: %u | Average Adjacency: %u\n", min_adjacency_length, max_adjacency_length, csr_mat.row_offsets[csr_mat.rows] / csr_mat.rows);
        
        // #######################################################
        // #######################################################
        // #######################################################
        // Testcase
        // #######################################################
        // #######################################################
        // #######################################################
        size_t allocationSize{4ULL * 1024ULL * 1024ULL * 1024ULL};
        #ifdef TEST_CUDA
        std::cout << "--- CUDA ---\n";
        DynGraph<VertexData, EdgeData, MemoryManagerCUDA> dynamic_graph(allocationSize);
        #elif TEST_HALLOC
        std::cout << "--- Halloc ---\n";
        DynGraph<VertexData, EdgeData, MemoryManagerHalloc> dynamic_graph(allocationSize);
        #elif TEST_SCATTERALLOC
        std::cout << "--- ScatterAlloc ---\n";
        DynGraph<VertexData, EdgeData, MemoryManagerScatterAlloc> dynamic_graph(allocationSize);
        #elif TEST_OUROBOROS
        std::cout << "--- Ouroboros ---";
        #ifdef TEST_PAGES
        #ifdef TEST_VIRTUALIZED_ARRAY
        std::cout << " Page --- Virtualized Array ---\n";
        DynGraph<VertexData, EdgeData, MemoryManagerOuroboros<OuroVAPQ>> dynamic_graph(allocationSize);
        #elif TEST_VIRTUALIZED_LIST
        std::cout << " Page --- Virtualized List ---\n";
        DynGraph<VertexData, EdgeData, MemoryManagerOuroboros<OuroVLPQ>> dynamic_graph(allocationSize);
        #else
        std::cout << " Page --- Standard ---\n";
        DynGraph<VertexData, EdgeData, MemoryManagerOuroboros<OuroPQ>> dynamic_graph(allocationSize);
        #endif
        #endif
        #ifdef TEST_CHUNKS
        #ifdef TEST_VIRTUALIZED_ARRAY
        std::cout << " Chunk --- Virtualized Array ---\n";
        DynGraph<VertexData, EdgeData, MemoryManagerOuroboros<OuroVACQ>> dynamic_graph(allocationSize);
        #elif TEST_VIRTUALIZED_LIST
        std::cout << " Chunk --- Virtualized List ---\n";
        DynGraph<VertexData, EdgeData, MemoryManagerOuroboros<OuroVLCQ>> dynamic_graph(allocationSize);
        #else
        std::cout << " Chunk --- Standard ---\n";
        DynGraph<VertexData, EdgeData, MemoryManagerOuroboros<OuroCQ>> dynamic_graph(allocationSize);
        #endif
        #endif
        #elif TEST_FDG
        std::cout << "--- FDGMalloc ---\n";
        DynGraph<VertexData, EdgeData, MemoryManagerFDG> dynamic_graph(allocationSize);
        #elif TEST_REGEFF
        std::cout << "--- RegEff ---";
        #ifdef TEST_ATOMIC
        std::cout << " Atomic\n";
        DynGraph<VertexData, EdgeData, MemoryManagerRegEff<RegEffVariants::AtomicMalloc>> dynamic_graph(allocationSize);
        #elif TEST_ATOMIC_WRAP
        std::cout << " Atomic Wrap\n";
        DynGraph<VertexData, EdgeData, MemoryManagerRegEff<RegEffVariants::AWMalloc>> dynamic_graph(allocationSize);
        #elif TEST_CIRCULAR
        std::cout << " Circular\n";
        DynGraph<VertexData, EdgeData, MemoryManagerRegEff<RegEffVariants::CMalloc>> dynamic_graph(allocationSize);
        #elif TEST_CIRCULAR_FUSED
        std::cout << " Circular Fused\n";
        DynGraph<VertexData, EdgeData, MemoryManagerRegEff<RegEffVariants::CFMalloc>> dynamic_graph(allocationSize);
        #elif TEST_CIRCULAR_MULTI
        std::cout << " Circular Multi\n";
        DynGraph<VertexData, EdgeData, MemoryManagerRegEff<RegEffVariants::CMMalloc>> dynamic_graph(allocationSize);
        #elif TEST_CIRCULAR_FUSED_MULTI
        std::cout << " Circular Fused Multi\n";
        DynGraph<VertexData, EdgeData, MemoryManagerRegEff<RegEffVariants::CFMMalloc>> dynamic_graph(allocationSize);
        #endif        
        #endif

        dynamic_graph.init(csr_mat);
    }
    
    return 0;
}