import os
import sys
import shutil
import time
from datetime import datetime
from timedprocess import Command

import argparse

def main():
	# Run all files from a directory
	print("##############################################################################")
	print("Callable as: python test_allocation.py")
	print("##############################################################################")
	
	# Specify which test configuration to use
	testcases = list()
	num_allocations = 10000
	smallest_allocation_size = 4
	largest_allocation_size = 1024
	free_memory = 1
	build_path = "../build/"

	parser = argparse.ArgumentParser(description='Test allocation performance for various frameworks')
	parser.add_argument('-t', type=str, help='Specify which frameworks to test, separated by +, e.g. o+s+h+c')
	parser.add_argument('-num', type=int, help='How many allocations to perform')
	parser.add_argument('-range', type=str, help='Sepcify Allocation Range, e.g. 4-1024')

	args = parser.parse_args()

	# Parse approaches
	if(args.t):
		selected_approaches = args.t.split('+')
		if any("h" in s for s in args.t):
			testcases.append(build_path + str("h_allocation"))
		if any("s" in s for s in args.t):
			testcases.append(build_path + str("s_allocation"))
		if any("o" in s for s in args.t):
			testcases.append(build_path + str("o_allocation"))
		if any("c" in s for s in args.t):
			testcases.append(build_path + str("c_allocation"))
	# Parse num allocation
	if(args.num):
		num_allocations = args.num

	# Parse range
	if(args.range):
		selected_range = args.range.split('-')
		smallest_allocation_size = int(selected_range[0])
		largest_allocation_size = int(selected_range[1])

	testcases.sort()

	print(testcases)

	exit()

	# Timeout (in seconds)
	time_out_val = 10;

	# Run executables
	for executable in testcases:
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