//------------------------------------------------------------------------------
// CSR.h
//
//
//------------------------------------------------------------------------------
//

#pragma once

#include <memory>
#include <algorithm>
#include <math.h>
#include <cstring>

template<typename T>
struct COO;

template<typename T>
struct DenseVector;

template<typename T>
struct CSR
{
	std::string filename;
	size_t rows, cols, nnz;

	std::unique_ptr<T[]> data;
	std::unique_ptr<unsigned int[]> row_offsets;
	std::unique_ptr<unsigned int[]> col_ids;

	CSR() : rows(0), cols(0), nnz(0) { }
	void alloc(size_t rows, size_t cols, size_t nnz, bool allocData=true);

	struct Statistics
	{
		double mean;
		double std_dev;
		size_t max;
		size_t min;
	};

	void computeStatistics(double& mean, double& std_dev, size_t& max, size_t& min)
	{
		// running variance by Welford
		size_t count = 0;
		mean = 0;
		double M2 = 0;
		max = 0;
		min = cols;
		for (size_t i = 0; i < rows; ++i)
		{
			size_t r_length = row_offsets[i + 1] - row_offsets[i];
			min = std::min(min, r_length);
			max = std::max(max, r_length);
			++count;
			double newValue = static_cast<double>(r_length);
			double delta = newValue - mean;
			mean = mean + delta / count;
			double delta2 = newValue - mean;
			M2 = M2 + delta * delta2;
		}
		if (count < 2)
			std_dev = 0;
		else
			std_dev = sqrt(M2 / (count - 1));
	}

	Statistics rowStatistics()
	{
		Statistics stats;
		computeStatistics(stats.mean, stats.std_dev, stats.max, stats.min);
		return stats;
	}
};


template<typename T>
CSR<T> loadCSR(const char* file);
template<typename T>
void storeCSR(const CSR<T>& mat, const char* file);

template<typename T>
void storeCSRStandardFormat(const CSR<T>& mat, const char* file);

template<typename T>
void spmv(DenseVector<T>& res, const CSR<T>& m, const DenseVector<T>& v, bool transpose = false);

template<typename T>
void convert(CSR<T>& res, const COO<T>& coo);
