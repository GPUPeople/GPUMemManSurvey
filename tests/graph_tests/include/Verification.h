#pragma once
#include "CSR.h"
#include "EdgeUpdate.h"

template <typename DataType>
struct Verification
{
	// ##############################################################################################################################################
	//
	Verification(const CSR<DataType>& input_graph) : offset_(input_graph.rows + 1), adjacency_(input_graph.rows), data_(input_graph.rows), 
	rows_(input_graph.rows), cols_(input_graph.cols), nnz_(input_graph.nnz)
	{
		memcpy(offset_.data(), input_graph.row_offsets.get(), sizeof(unsigned int) * (input_graph.rows + 1));
		for(auto i = 0; i < rows_; ++i)
		{
			auto number_neighbours = input_graph.row_offsets[i + 1] - input_graph.row_offsets[i];

			adjacency_[i].resize(number_neighbours);
			data_[i].resize(number_neighbours);

			memcpy(adjacency_[i].data(), &input_graph.col_ids[input_graph.row_offsets[i]], sizeof(unsigned int) * number_neighbours);
			memcpy(data_[i].data(), &input_graph.data[input_graph.row_offsets[i]], sizeof(DataType) * number_neighbours);
		}
	}

	// ##############################################################################################################################################
	//
	template <typename VertexDataType, typename EdgeDataType>
	void hostEdgeInsertion(EdgeUpdateBatch<VertexDataType, EdgeDataType>& updates)
	{
		auto batch_size = updates.edge_update.size();
		auto duplicates = 0U;
		for (auto i = 0U; i < batch_size; ++i)
		{

			auto percentage = ((static_cast<double>(i) * 100) / batch_size) / 100.0;
			printProgressBar(percentage);

			auto edge_src = updates.edge_update[i].source;
			auto edge_dest = updates.edge_update[i].update.destination;

			// TODO: Currently no support here for adding new vertices
			if (edge_src >= rows_ || edge_dest >= rows_)
				continue;

			auto num_neighbours = offset_[edge_src + 1] - offset_[edge_src];
			auto& adjacency = adjacency_.at(edge_src);
			auto& data = data_.at(edge_src);

			if(std::find(adjacency.begin(), adjacency.end(), edge_dest) == adjacency.end())
			{
				// Not in the adjacency, insert
				adjacency.push_back(edge_dest);
				for(auto j = edge_src + 1; j < rows_ + 1; ++j)
				{
					offset_[j] += 1;
				}
				++nnz_;
			}
			else
			{
				++duplicates;
			}
			
		}
		printProgressBarEnd();
		// printf("Duplicate on host: %u\n", duplicates);
	}

	// ##############################################################################################################################################
	//
	template <typename VertexDataType, typename EdgeDataType>
	void hostEdgeDeletion(EdgeUpdateBatch<VertexDataType, EdgeDataType>& updates)
	{
		auto batch_size = updates.edge_update.size();

		for (auto i = 0U; i < batch_size; ++i)
		{
			auto percentage = ((static_cast<double>(i) * 100) / batch_size) / 100.0;
			printProgressBar(percentage);
			auto edge_src = updates.edge_update[i].source;
			auto edge_dest = updates.edge_update[i].update.destination;

			if (edge_src >= rows_ || edge_dest >= rows_)
				continue;

			auto num_neighbours = offset_[edge_src + 1] - offset_[edge_src];
			auto& adjacency = adjacency_.at(edge_src);
			auto& data = data_.at(edge_src);

			 auto pos = std::find(adjacency.begin(), adjacency.end(), edge_dest);
			if(pos != adjacency.end())
			{
				// Found edge, will be deleted now
				adjacency.erase(pos); 
				for (auto j = edge_src + 1; j < rows_ + 1; ++j)
				{
					offset_[j] -= 1;
				}
				--nnz_;
			}
		}
		printProgressBarEnd();
	}

	// ##############################################################################################################################################
	//
	bool verify(const CSR<DataType>& dynGraph, const char* header, OutputCodes output_code)
	{
		auto correct{ true };
		if (nnz_ != dynGraph.nnz)
		{
			printf("%sError - %s\n%s", CLHighlight::break_line_red_s, header, CLHighlight::break_line_red_e);
			printf("NNZ missing: Ref: %u | Dyn: %u\n", nnz_, dynGraph.nnz);
			printf("%s", CLHighlight::break_line_red);
			correct = false;
		}
		else
		{
			for (auto i = 0; i < rows_; ++i)
			{
				auto offset = offset_[i];
				auto neighbours = offset_[i + 1] - offset;
				if (offset != dynGraph.row_offsets[i])
				{
					printf("%sError - %s\n%s", CLHighlight::break_line_red_s, header, CLHighlight::break_line_red_e);
					printf("Row %d : ref:[%u] dyn:[%u] | Neighbours wrong!\n", i, offset_[i], dynGraph.row_offsets[i]);
					printf("%s", CLHighlight::break_line_red);
					correct = false;
					break;
				}
				else
				{
					auto& adjacency = adjacency_.at(i);
					for (int j = 0; offset < offset_[i + 1]; ++offset, ++j)
						{
						bool edge_found{ false };
						for (auto k = dynGraph.row_offsets[i]; k < dynGraph.row_offsets[i + 1]; ++k)
						{
							if (adjacency[j] == dynGraph.col_ids[k])
								edge_found = true;
						}
						if (!edge_found)
						{
							printf("%sError - %s\n%s", CLHighlight::break_line_red_s, header, CLHighlight::break_line_red_e);
							printf("Row %d : Offset: %u | ref:[%u] Edge NOT found!\n", i, offset, adjacency[j]);
							printf("------------------------------------------------\n");
							printf("Neighbours: %u vs. %u\n", neighbours, dynGraph.row_offsets[i + 1] - dynGraph.row_offsets[i]);
							printf("Reference\n");
							for (auto k = 0; k < neighbours; ++k)
							{
								printf("%u | ", adjacency[k]);
							}
							printf("\nDynGraph:\n");
							for (auto k = dynGraph.row_offsets[i]; k < dynGraph.row_offsets[i + 1]; ++k)
							{
								printf("%u | ", dynGraph.col_ids[k]);
							}
							printf("\n");
							printf("%s", CLHighlight::break_line_red);
							correct = false;
							break;
						}
					}
				}
				if (!correct)
					break;
			}
		}
		if(!correct)
		{
			std::ofstream res_file;
			res_file.open("res.txt", std::ofstream::out);
			if (res_file.is_open())
			{
				res_file << static_cast<int>(output_code) << "\n";
				res_file.close();
			}
			exit(-1);
		}
		else
		{
			printf("%sVerification done - %s\n%s", CLHighlight::break_line_green_s, header, CLHighlight::break_line);
			printf("Graph Stats: Number Vertices: %llu | Number Edges: %llu\n", rows_, nnz_);
			printf("%s", CLHighlight::break_line_green_e);
			return true;
		}
	}

	void printAdjacency(unsigned int vertex_index)
	{
		auto offset = offset_[vertex_index];
		auto neighbours = offset_[vertex_index + 1] - offset;
		for(auto i = 0; i < neighbours; ++i)
		{
			std::cout << adjacency_[vertex_index][i] << " | ";
		}
		std::cout << std::endl;
	}

	size_t rows_;
	size_t cols_;
	size_t nnz_;
	std::vector<unsigned int> offset_;
	std::vector<std::vector<unsigned int>> adjacency_;
	std::vector<std::vector<DataType>> data_;
};
