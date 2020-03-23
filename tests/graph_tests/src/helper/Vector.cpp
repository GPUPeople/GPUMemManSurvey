#include "Vector.h"

#include "cuda_runtime_api.h"

namespace
{
	template<typename DataType>
	void dealloc(dDenseVector<DataType>& vec)
	{
		if (vec.data != nullptr)
			cudaFree(vec.data);
		vec.data = nullptr;
		vec.size = 0;
	}
}

template <typename DataType>
void dDenseVector<DataType>::alloc(size_t elements)
{
	if(data != nullptr)
	{
		cudaMalloc(reinterpret_cast<void**>(&data), sizeof(DataType) * elements);
		size = elements;
	}
}

template <typename DataType>
void dDenseVector<DataType>::reset()
{
	dealloc(*this);
}


template<typename DataType>
dDenseVector<DataType>::~dDenseVector()
{
	dealloc(*this);
}

template<typename T>
void convert(dDenseVector<T>& dst, const DenseVector<T>& src)
{
	dst.alloc(src.size);
	cudaMemcpy(dst.data, &src.data[0], src.size * sizeof(T), cudaMemcpyHostToDevice);
}

template<typename T>
void convert(DenseVector<T>& dst, const dDenseVector<T>& src)
{
	if(dst.size != src.size)
		dst.alloc(src.size);
	cudaMemcpy(dst.data.get(), &src.data[0], src.size * sizeof(T), cudaMemcpyDeviceToHost);
}


// Explicit instantiations
template void convert(DenseVector<float>&, const dDenseVector<float>&);
template void convert(DenseVector<double>&, const dDenseVector<double>&);

template void convert(dDenseVector<float>&, const DenseVector<float>&);
template void convert(dDenseVector<double>&, const DenseVector<double>&);

template dDenseVector<float>::~dDenseVector();
template dDenseVector<double>::~dDenseVector();

template void dDenseVector<float>::reset();
template void dDenseVector<double>::reset();

template void dDenseVector<float>::alloc(size_t elements);
template void dDenseVector<double>::alloc(size_t elements);