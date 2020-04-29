#pragma once

#include <vector>
#include <numeric>
#include <algorithm>
#include "UtilityFunctions.cuh"

struct DevicePerf
{
    using Type = long long int;
    Type start_cycle{0};
    
    __forceinline__ __device__ void startThreadMeasure() 
    {
        #ifdef __CUDA_ARCH__
        start_cycle = clock64();
        #endif
    }
    __forceinline__ __device__ Type stopThreadMeasure() 
    {
        #ifdef __CUDA_ARCH__
        return clock64() - start_cycle;
        #else
        return 0LLU;
        #endif
    }
};

struct ThreadResult
{
    using Type = DevicePerf::Type;

	double mean_{0.0};
	double std_dev_{0.0};
    double median_{0.0};
    double min_{0.0};
    double max_{0.0};
	int num_{0};
};

struct DevicePerfMeasure
{
    using Type = DevicePerf::Type;
    std::vector<Type> perf_;
    std::vector<double> results;
    Type* d_perf_;
    unsigned int iter{0};
    unsigned int num_threads{0};

    DevicePerfMeasure(unsigned int num_threads, unsigned int iterations) : num_threads{num_threads}
    {
        CHECK_ERROR(cudaMalloc(&d_perf_, sizeof(Type) * num_threads));
        CHECK_ERROR(cudaMemset(d_perf_, 0, sizeof(Type) * num_threads));
        perf_.resize(num_threads * iterations);
        results.resize(num_threads * iterations);
    }

    ~DevicePerfMeasure()
    {
        CHECK_ERROR(cudaFree(d_perf_));
    }

    void acceptResultsFromDevice()
    {
        CHECK_ERROR(cudaMemcpy(&perf_[iter*num_threads], d_perf_, sizeof(Type) * num_threads, cudaMemcpyDeviceToHost));
        CHECK_ERROR(cudaMemset(d_perf_, 0, sizeof(Type) * num_threads));
        ++iter;
    }

    Type* getDevicePtr(){return d_perf_;}

    double mean()
	{
		return std::accumulate(results.begin(), results.end(), 0.0) / static_cast<double>(perf_.size());
    }
    
    double median()
	{
        std::sort(results.begin(), results.end());
		return results[results.size() / 2];
    }
    
    double std_dev(double mean)
	{
		for(auto& elem : results)
			elem = (elem - mean)*(elem - mean);
		return sqrt(std::accumulate(results.begin(), results.end(), 0.0) / results.size());
	}

    ThreadResult generateResult()
    {
        std::copy(perf_.begin(), perf_.end(), results.begin());
        auto mean_ = mean();
        return ThreadResult{
            mean_,
            std_dev(mean_),
            median(),
            *std::min_element(results.begin(), results.end()),
			*std::max_element(results.begin(), results.end()),
            static_cast<int>(perf_.size())
        };
    }
};