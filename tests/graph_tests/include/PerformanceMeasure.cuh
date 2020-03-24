#pragma once

#include <vector>
#include <numeric>
#include "UtilityFunctions.cuh"

struct Result
{
	float mean_{0.0f};
	float median_{0.0f};
	int num{0};
};

struct PerfMeasure
{
	cudaEvent_t ce_start, ce_stop;
	std::vector<float> measurements_;

	void startMeasurement(){Utils::start_clock(ce_start, ce_stop);}
	void stopMeasurement(){measurements_.push_back(Utils::end_clock(ce_start, ce_stop));}
	float mean()
	{
		return std::accumulate(measurements_.begin(), measurements_.end(), 0.0f) / static_cast<float>(measurements_.size());
	}
	float median()
	{
		std::vector<float> sorted_measurements(measurements_);
		return sorted_measurements[sorted_measurements.size() / 2];
	}
	Result generateResult()
	{
		return Result{
			mean(),
			median(),
			static_cast<int>(measurements_.size())
		};
	}
};