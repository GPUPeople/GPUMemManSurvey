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

	double mean_{0.0f};
	double std_dev_{0.0f};
	Type median_{0LLU};
	int num_{0};
};

struct DevicePerfMeasure
{
    using Type = DevicePerf::Type;
    std::vector<Type> perf_;
    Type* d_perf_;

    DevicePerfMeasure(unsigned int num_threads)
    {
        CHECK_ERROR(cudaMalloc(&d_perf_, sizeof(Type) * num_threads));
        perf_.reserve(num_threads);
    }

    Type* getDevicePtr(){return d_perf_;}

    double mean()
	{
		return std::accumulate(perf_.begin(), perf_.end(), 0.0f) / static_cast<double>(perf_.size());
    }
    Type median()
	{
        std::vector<Type> sorted_measurements(perf_);
        std::sort(sorted_measurements.begin(), sorted_measurements.end());
		return sorted_measurements[sorted_measurements.size() / 2];
    }
    
    double std_dev(double mean)
	{
		std::vector<double> stdmean_measurements(perf_.begin(), perf_.end());
		for(auto& elem : stdmean_measurements)
			elem = (elem - mean)*(elem - mean);
		return sqrt(std::accumulate(stdmean_measurements.begin(), stdmean_measurements.end(), 0.0f) / static_cast<double>(stdmean_measurements.size()));
	}

    ThreadResult computeResult()
    {
        CHECK_ERROR(cudaMemcpy(perf_.data(), d_perf_, sizeof(Type) * perf_.size(), cudaMemcpyDeviceToHost));
        double mean_ = mean();
        return ThreadResult{
            mean_,
            std_dev(mean_),
            median(),
            static_cast<int>(perf_.size())
        };
    }
};