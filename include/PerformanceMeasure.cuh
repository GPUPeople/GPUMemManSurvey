#pragma once

#include <vector>
#include <numeric>
#include "UtilityFunctions.cuh"

struct Result
{
	float mean_{0.0f};
	float std_dev_{0.0f};
	float median_{0.0f};
	int num_{0};
};

struct PerfMeasure
{
	cudaEvent_t ce_start, ce_stop;
	std::vector<float> measurements_;

	void startMeasurement(){Utils::start_clock(ce_start, ce_stop);}
	void stopMeasurement(){measurements_.push_back(Utils::end_clock(ce_start, ce_stop));}
	void addMeasure(PerfMeasure& measure){measurements_.insert(measurements_.end(), measure.measurements_.begin(), measure.measurements_.end());}
	
	float mean()
	{
		return std::accumulate(measurements_.begin(), measurements_.end(), 0.0f) / static_cast<float>(measurements_.size());
	}
	float median()
	{
		std::vector<float> sorted_measurements(measurements_);
		return sorted_measurements[sorted_measurements.size() / 2];
	}
	float std_dev(float mean)
	{
		std::vector<float> stdmean_measurements(measurements_);
		for(auto& elem : stdmean_measurements)
			elem = (elem - mean)*(elem - mean);
		return sqrt(std::accumulate(measurements_.begin(), measurements_.end(), 0.0f) / static_cast<float>(measurements_.size()));
	}

	Result generateResult()
	{
		float mean_val = mean();
		return Result{
			mean_val,
			std_dev(mean_val),
			median(),
			static_cast<int>(measurements_.size())
		};
	}
};