//------------------------------------------------------------------------------
// Vector.h
//
//
//------------------------------------------------------------------------------
//

#pragma once

#include <memory>

template<typename T>
struct DenseVector
{
	size_t size;
	std::unique_ptr<T[]> data;

	DenseVector() : size(0) { }
	void alloc(size_t s)
	{
		data = std::make_unique<T[]>(s);
		size = s;
	}
};

template<typename T>
struct dDenseVector
{
	size_t size;
	T* data;

	dDenseVector() : size(0), data{nullptr} { }
	void alloc(size_t elements);
	void reset();
	~dDenseVector();
};

template<typename T>
void convert(dDenseVector<T>& dst, const DenseVector<T>& src);

template<typename T>
void convert(DenseVector<T>& dst, const dDenseVector<T>& src);