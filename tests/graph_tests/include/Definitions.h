//------------------------------------------------------------------------------
// Definitions.h
//
//
//------------------------------------------------------------------------------
//

#pragma once

#include <typeinfo>
#include <memory>
#include <vector>
#include <limits>
#include <cuda_runtime_api.h>
#include <device_launch_parameters.h>
#include <cuda_runtime.h>

//------------------------------------------------------------------------------
// Datatype definitions
//------------------------------------------------------------------------------
using vertex_t = uint32_t;
using index_t = vertex_t;
using memory_t = int8_t;
using matrix_t = uint32_t;
using OffsetList_t = std::vector<vertex_t>;
using AdjacencyList_t = std::vector<vertex_t>;
using MatrixList_t = std::vector<matrix_t>;

//------------------------------------------------------------------------------
// Helper definitions
//------------------------------------------------------------------------------
#define SINGLE_THREAD (threadIdx.x == 0)
#define SINGLE_THREAD_MULTI (threadID == 0)

static constexpr bool THRUST_SORT{true};

static constexpr index_t DELETIONMARKER{ std::numeric_limits<index_t>::max() };

// If edge insertions should also do updating when collision happens
static constexpr bool updateValues{false};

static constexpr int minPageSize{16}; // Smallest Page Size is 16 Bytes

static constexpr bool realistic_deletion{true};

static constexpr bool printDebugMessages{true};

enum class OutputCodes : int
{
	OKAY = 0,
	VERIFY_INITIALIZATION = 1,
	VERIFY_INSERTION = 2,
	VERIFY_DELETION = 3
};

namespace CLHighlight
{
	static constexpr char break_line[] = {"##########################################################################################################\n"};
	static constexpr char break_line_red[] = {"\033[0;31m##########################################################################################################\033[0m\n"};
	static constexpr char break_line_red_s[] = {"\033[0;31m##########################################################################################################\n"};
	static constexpr char break_line_red_e[] = {"##########################################################################################################\033[0m\n"};
	static constexpr char break_line_green[] = {"\033[0;32m##########################################################################################################\033[0m\n"};
	static constexpr char break_line_green_s[] = {"\033[0;32m##########################################################################################################\n"};
	static constexpr char break_line_green_e[] = {"##########################################################################################################\033[0m\n"};
	static constexpr char break_line_blue[] = {"\033[0;34m##########################################################################################################\033[0m\n"};
	static constexpr char break_line_blue_s[] = {"\033[0;34m##########################################################################################################\n"};
	static constexpr char break_line_blue_e[] = {"##########################################################################################################\033[0m\n"};
	static constexpr char break_line_purple[] = {"\033[0;35m##########################################################################################################\033[0m\n"};
	static constexpr char break_line_purple_s[] = {"\033[0;35m##########################################################################################################\n"};
	static constexpr char break_line_purple_e[] = {"##########################################################################################################\033[0m\n"};
	static constexpr char break_line_lblue[] = {"\033[1;34m##########################################################################################################\033[0m\n"};
	static constexpr char break_line_cyan[] = {"\033[0;36m##########################################################################################################\033[0m\n"};
	static constexpr char break_line_cyan_s[] = {"\033[0;36m##########################################################################################################\n"};
	static constexpr char break_line_cyan_e[] = {"##########################################################################################################\033[0m\n"};
}


