import os
import sys
import shutil
import time
from timedprocess import Command


def main():
	# Run all files from a directory
	print("##############################################################################")
	print("Callable as: python testallocations.py")
	print("##############################################################################")

	# Specify build folders here
	build_paths = {
		 "../allocations/build/",
		"../allocations/async_build/"
	}
	# Specify which tests to test
	testcases = {

		# Array-based

		"test_large_allocation_p",
		"test_many_small_allocations_p",
		"test_many_larger_allocations_p",
		"test_frees_after_large_allocation_p",
		"test_mixed_allocation_p",

		"test_large_allocation_c",
		"test_many_small_allocations_c",
		"test_many_larger_allocations_c",
		"test_frees_after_large_allocation_c",
		"test_mixed_allocation_c",


		# Virtualized Array-based 

		"v_test_large_allocation_p",
		"v_test_many_small_allocations_p",
		"v_test_many_larger_allocations_p",
		"v_test_frees_after_large_allocation_p",
		"v_test_mixed_allocation_p",

		"v_test_large_allocation_c",
		"v_test_many_small_allocations_c",
		"v_test_many_larger_allocations_c",
		"v_test_frees_after_large_allocation_c",
		"v_test_mixed_allocation_c",


		# Virtualized Linked-based 

		"vl_test_large_allocation_p",
		"vl_test_many_small_allocations_p",
		"vl_test_many_larger_allocations_p",
		"vl_test_frees_after_large_allocation_p",
		"vl_test_mixed_allocation_p",

		"vl_test_large_allocation_c",
		"vl_test_many_small_allocations_c",
		"vl_test_many_larger_allocations_c",
		"vl_test_frees_after_large_allocation_c",
		"vl_test_mixed_allocation_c",


		# CUDA

		# "c_test_large_allocation",
		# "c_test_many_small_allocations",
		# "c_test_many_larger_allocations",
		# "c_test_frees_after_large_allocation",
		# "c_test_mixed_allocation"
	}
	test_paths = list()
	for build_path in build_paths:
		for testcase in testcases:
			test_paths.append(build_path + testcase)

	test_paths.sort()

	# Specify which test configuration to use
	run_config = "test_allocation.json"

	# Timeout (in seconds)
	time_out_val = 3600 * 1;

	for executable in test_paths:
		executecommand = "{0} {1}".format(executable, run_config)
		print(executecommand)
		Command(executecommand).run(timeout=time_out_val)
		print("Test done")
	
	print("Tests done")


if __name__ == "__main__":
	main()