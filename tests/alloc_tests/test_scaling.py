import sys
# the mock-0.3.1 dir contains testcase.py, testutils.py & mock.py
sys.path.append('../../scripts')

import os
import shutil
import time
from datetime import datetime
from timedprocess import Command
import pandas
import numpy as np
import csv
import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use('seaborn-whitegrid')

colours = {
	'Halloc' : 'orange' , 
	'Ouroboros-P-VA' : 'lightcoral' , 'Ouroboros-P-VL' : 'darkred' , 'Ouroboros-P-S' : 'red' ,
	'Ouroboros-C-VA' : 'red' , 'Ouroboros-C-VL' : 'red' , 'Ouroboros-C-S' : 'red' ,
	'CUDA' : 'green' , 
	'ScatterAlloc' : 'blue' , 
	'FDGMalloc' : 'gold' , 
	'RegEff-A' : 'mediumvioletred' , 'RegEff-AW' : 'orchid',
	'RegEff-C' : 'purple' , 'RegEff-CF' : 'violet' , 'RegEff-CM' : 'indigo' , 'RegEff-CFM' : 'blueviolet'
}

import argparse

def main():
	# Run all files from a directory
	print("##############################################################################")
	print("Callable as: python test_scaling.py -h")
	print("##############################################################################")
	
	# Specify which test configuration to use
	testcases = list()
	smallest_allocation_size = 4
	largest_allocation_size = 1024
	smallest_num_threads = 2 ** 0
	largest_num_threads = 2 ** 10
	num_iterations = 25
	free_memory = 1
	build_path = "build/"

	parser = argparse.ArgumentParser(description='Test allocation performance for various frameworks')
	parser.add_argument('-t', type=str, help='Specify which frameworks to test, separated by +, e.g. o+s+h+c+f+r ---> c : cuda | s : scatteralloc | h : halloc | o : ouroboros | f : fdgmalloc | r : register-efficient')
	parser.add_argument('-byterange', type=str, help='Specify Allocation Range, given as powers of two, e.g. 0-5 -> results in 1-32')
	parser.add_argument('-threadrange', type=str, help='Specify Allocation Range, given as powers of two, e.g. 0-5 -> results in 1-32')
	parser.add_argument('-iter', type=int, help='How many iterations?')
	parser.add_argument('-runtest', action='store_true', default=False, help='Run testcases')
	parser.add_argument('-genres', action='store_true', default=False, help='Generate results')
	parser.add_argument('-genplot', action='store_true', default=False, help='Generate results file and plot')
	parser.add_argument('-cleantemp', action='store_true', default=False, help='Clean up temporary files')
	parser.add_argument('-warp', action='store_true', default=False, help='Start testcases warp-based')
	parser.add_argument('-devmeasure', action='store_true', default=False, help='Measure performance on device in cycles')
	parser.add_argument('-plotscale', type=str, help='log/linear')

	args = parser.parse_args()

	# Parse approaches
	if(args.t):
		if any("h" in s for s in args.t):
			testcases.append(build_path + str("h_alloc_test"))
		if any("s" in s for s in args.t):
			testcases.append(build_path + str("s_alloc_test"))
		if any("o" in s for s in args.t):
			testcases.append(build_path + str("o_alloc_test_p"))
			testcases.append(build_path + str("o_alloc_test_c"))
			testcases.append(build_path + str("o_alloc_test_vap"))
			testcases.append(build_path + str("o_alloc_test_vac"))
			testcases.append(build_path + str("o_alloc_test_vlp"))
			testcases.append(build_path + str("o_alloc_test_vlc"))
		if any("c" in s for s in args.t):
			testcases.append(build_path + str("c_alloc_test"))
		if any("f" in s for s in args.t):
			testcases.append(build_path + str("f_alloc_test"))
		if any("r" in s for s in args.t):
			testcases.append(build_path + str("r_alloc_test_a"))
			testcases.append(build_path + str("r_alloc_test_aw"))
			# testcases.append(build_path + str("r_alloc_test_c"))
			testcases.append(build_path + str("r_alloc_test_cf"))
			# testcases.append(build_path + str("r_alloc_test_cm"))
			testcases.append(build_path + str("r_alloc_test_cfm"))
	
	# Parse allocation size
	if(args.byterange):
		selected_range = args.range.split('-')
		smallest_allocation_size = 2 ** int(selected_range[0])
		largest_allocation_size = 2 ** int(selected_range[1])

	# Parse range
	if(args.threadrange):
		selected_range = args.range.split('-')
		smallest_num_threads = 2 ** int(selected_range[0])
		largest_num_threads = 2 ** int(selected_range[1])

	# Parse num iterations
	if(args.iter):
		num_iterations = args.iter

	# Generate results
	if args.warp:
		test_warp_based = 1
	else:
		test_warp_based = 0
	
	# Run Testcases
	run_testcases = args.runtest
	
	# Generate results
	generate_results = args.genres

	# Generate plots
	generate_plots = args.genplot

	# Clean temporary files
	clean_temporary_files = args.cleantemp

	# Measure on device
	if args.devmeasure:
		measure_on_device = 1
	else:
		measure_on_device = 0

	if measure_on_device and test_warp_based:
		print("Cannot measure on device and warp-based at the same time!")
		exit(-1)

	testcases.sort()

	# Timeout (in seconds)
	time_out_val = 10

	####################################################################################################
	####################################################################################################
	# Run testcases
	####################################################################################################
	####################################################################################################
	if run_testcases:
		for executable in testcases:
			allocation_size = smallest_allocation_size
			while allocation_size <= largest_allocation_size:
				num_threads = smallest_num_threads
				while num_threads <= largest_num_threads:
					run_config = str(smallest_num_threads) + " " + str(smallest_allocation_size) + " " + str(num_iterations) + " " + str(measure_on_device) + " " + str(test_warp_based) + " 1 " + str(free_memory) + " results/tmp/"
					executecommand = "{0} {1}".format(executable, run_config)
					print(executecommand)
					Command(executecommand).run(timeout=time_out_val)
					num_threads *= 2
				allocation_size *= 2

	####################################################################################################
	####################################################################################################
	# Generate new Results
	####################################################################################################
	####################################################################################################
	if generate_results:
		print("Generalte results")

	####################################################################################################
	####################################################################################################
	# Generate plots
	####################################################################################################
	####################################################################################################
	if generate_plots:
		print("Generalte plots")

	####################################################################################################
	####################################################################################################
	# Clean temporary files
	####################################################################################################
	####################################################################################################
	if clean_temporary_files:
		for file in os.listdir("results/tmp"):
			filename = str("results/tmp/") + os.fsdecode(file)
			if(os.path.isdir(filename)):
				continue
			os.remove(filename)
		for file in os.listdir("results/tmp/aggregate"):
			filename = str("results/tmp/aggregate/") + os.fsdecode(file)
			if(os.path.isdir(filename)):
				continue
			os.remove(filename)
		for file in os.listdir("results/plots"):
			filename = str("results/plots/") + os.fsdecode(file)
			if(os.path.isdir(filename)):
				continue
			os.remove(filename)

	print("Done")

if __name__ == "__main__":
	main()