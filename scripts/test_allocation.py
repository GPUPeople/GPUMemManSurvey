import os
import sys
import shutil
import time
from datetime import datetime
from timedprocess import Command

def main():
	# Run all files from a directory
	print("##############################################################################")
	print("Callable as: python test_allocation.py")
	print("##############################################################################")
	
	build_path = "../build/"
	testcases = {
		 "c_allocation",
		 "h_allocation",
		 "o_allocation",
		 "s_allocation"
	}

	test_paths = list()
	for testcase in testcases:
		test_paths.append(build_path + testcase)

	test_paths.sort()

	print(test_paths)

	# Specify which test configuration to use
	num_allocations = 10000
	smallest_allocation_size = 4
	largest_allocation_size = 1024
	free_memory = 1

	# Timeout (in seconds)
	time_out_val = 10;

	# Run executables
	for executable in test_paths:
		smallest_allocation_size = 4
		while smallest_allocation_size < largest_allocation_size:
			run_config = str(num_allocations) + " " + str(smallest_allocation_size) + " 0 " + str(free_memory)
			executecommand = "{0} {1}".format(executable, run_config)
			print(executecommand)
			Command(executecommand).run(timeout=time_out_val)
			smallest_allocation_size += 4
	
	# Copy resulting files into separate folder with time stamp
	now = datetime.now()
	time_string = now.strftime("%b-%d-%Y_%H-%M-%S")
	for file in os.listdir("../results"):
		filename = fixed_files_path + time_string + os.fsdecode(file)
		shutil.move(filename, "../results/allocation")


if __name__ == "__main__":
	main()