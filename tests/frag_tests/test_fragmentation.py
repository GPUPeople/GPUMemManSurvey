import os
import sys
import shutil
import time
from datetime import datetime
from timedprocess import Command
import pandas
import numpy as np
import csv
import matplotlib.pyplot as plt
plt.style.use('seaborn-whitegrid')

colours = {'Actual Size' : 'Black', 'Halloc' : 'Orange' , 'Ouroboros' : 'Red' , 'CUDA' : 'green' , 'ScatterAlloc' : 'blue'}

import argparse

def main():
	# Run all files from a directory
	print("##############################################################################")
	print("Callable as: python test_fragmentation.py")
	print("##############################################################################")
	
	# Specify which test configuration to use
	testcases = {}
	num_allocations = 10000
	smallest_allocation_size = 4
	largest_allocation_size = 1024
	num_iterations = 1
	free_memory = 1
	generate_results = True
	generate_plots = True
	clean_temporary_files = True
	test_warp_based = False
	build_path = "../build/"

	parser = argparse.ArgumentParser(description='Test fragmentation for various frameworks')
	parser.add_argument('-t', type=str, help='Specify which frameworks to test, separated by +, e.g. o+s+h+c')
	parser.add_argument('-num', type=int, help='How many allocations to perform')
	parser.add_argument('-range', type=str, help='Sepcify Allocation Range, e.g. 4-1024')
	parser.add_argument('-iter', type=int, help='How many iterations?')
	parser.add_argument('-genplot', action='store_true', default=False, help='Generate results file and plot')
	parser.add_argument('-genres', action='store_true', default=False, help='Run testcases and generate results')
	parser.add_argument('-cleantemp', action='store_true', default=False, help='Clean up temporary files')
	parser.add_argument('-warp', action='store_true', default=False, help='Start testcases warp-based')

	args = parser.parse_args()

	# Parse approaches
	if(args.t):
		if any("c" in s for s in args.t):
			testcases["CUDA"] = build_path + str("c_alloc_test")
		if any("x" in s for s in args.t):
			testcases["XMalloc"] = build_path + str("x_alloc_test")
		if any("h" in s for s in args.t):
			testcases["Halloc"] = build_path + str("h_alloc_test")
		if any("s" in s for s in args.t):
			testcases["ScatterAlloc"] = sync_build_path + str("s_alloc_test")
		if any("o" in s for s in args.t):
			testcases["Ouroboros-P-S"] = build_path + str("o_alloc_test_p")
			testcases["Ouroboros-C-S"] = build_path + str("o_alloc_test_c")
			testcases["Ouroboros-P-VA"] = build_path + str("o_alloc_test_vap")
			testcases["Ouroboros-C-VA"] = build_path + str("o_alloc_test_vac")
			testcases["Ouroboros-P-VL"] = build_path + str("o_alloc_test_vlp")
			testcases["Ouroboros-C-VL"] = build_path + str("o_alloc_test_vlc")
		if any("f" in s for s in args.t):
			testcases["FDGMalloc"] = build_path + str("f_alloc_test")
		if any("r" in s for s in args.t):
			testcases["RegEff-A"] = build_path + str("r_alloc_test_a")
			testcases["RegEff-AW"] = build_path + str("r_alloc_test_aw")
			# testcases["RegEff-C"] = build_path + str("r_alloc_test_c")
			testcases["RegEff-CF"] = build_path + str("r_alloc_test_cf")
			# testcases["RegEff-CM"] = build_path + str("r_alloc_test_cm")
			testcases["RegEff-CFM"] = build_path + str("r_alloc_test_cfm")
	
	# Parse num allocation
	if(args.num):
		num_allocations = args.num

	# Parse range
	if(args.range):
		selected_range = args.range.split('-')
		smallest_allocation_size = int(selected_range[0])
		largest_allocation_size = int(selected_range[1])
	
	# Parse num iterations
	if(args.iter):
		num_iterations = args.iter

	# Generate results
	test_warp_based = args.warp
	
	# Generate results
	generate_results = args.genres

	# Generate plots
	generate_plots = args.genplot

	# Generate plots
	clean_temporary_files = args.cleantemp

	# Timeout (in seconds)
	time_out_val = 5

	####################################################################################################
	####################################################################################################
	# Generate new Results
	####################################################################################################
	####################################################################################################
	if generate_results:
		for _, executable in testcases.items():
			smallest_allocation_size = 4
			while smallest_allocation_size <= largest_allocation_size:
				warp_based = 0
				if test_warp_based:
					warp_based = 1
				run_config = str(num_allocations) + " " + str(smallest_allocation_size) + " " + str(num_iterations) + " " + str(warp_based) + " 0 " + str(free_memory)
				executecommand = "{0} {1}".format(executable, run_config)
				print(executecommand)
				Command(executecommand).run(timeout=time_out_val)
				smallest_allocation_size += 4

	####################################################################################################
	####################################################################################################
	# Generate new plots
	####################################################################################################
	####################################################################################################
	if generate_plots:
		approach_result_frag = list(list())
		approach_result_frag.append(np.arange(smallest_allocation_size, largest_allocation_size, 4).tolist())
		approach_result_frag[0].insert(0, "Bytes")
		approach_result_frag.append(np.arange(smallest_allocation_size * num_allocations, largest_allocation_size * num_allocations, 4 * num_allocations).tolist())
		approach_result_frag[1].insert(0, "Actual Size")

		# Go over files, read data and generate new 
		for file in os.listdir("../results"):
			filename = str("../results/") + os.fsdecode(file)
			if(os.path.isdir(filename)):
				continue
			with open(filename, newline='') as csv_file:
				dataframe = pandas.read_csv(csv_file)
				approach_result_frag.append(list(dataframe.iloc[:, 2]))
				approach_result_frag[-1].insert(0, os.fsdecode(file).split('_')[2])

		# Get Timestring
		now = datetime.now()
		time_string = now.strftime("%b-%d-%Y_%H-%M-%S")

		# Generate output file
		alloc_name = str("../results/fragmentation/") + time_string + str(num_allocations) + str(".csv")
		with(open(alloc_name, "w")) as f:
			writer = csv.writer(f, delimiter=',')
			for row in approach_result_frag:
				writer.writerow(row)

		# Generate output plot
		df = pandas.DataFrame({str(approach_result_frag[0][0]) : approach_result_frag[0][1:]})
		for i in range(1, len(approach_result_frag)):
			df[str(approach_result_frag[i][0])] = approach_result_frag[i][1:]

		for i in range(1, len(approach_result_frag)):
			plt.plot(str(approach_result_frag[0][0]), str(approach_result_frag[i][0]), data=df, marker='', color=colours[str(approach_result_frag[i][0])], linewidth=1, label=str(approach_result_frag[i][0]))
		plt.yscale("log")
		plt.ylabel('Bytes')
		plt.xlabel('Bytes')
		plt.title("Allocation Byte Range for " + str(num_allocations) + " allocations")
		plt.legend()
		plt.savefig(str("../results/fragmentation/") + time_string + "_frag.pdf", dpi=600)

	####################################################################################################
	####################################################################################################
	# Clean temporary files
	####################################################################################################
	####################################################################################################
	if clean_temporary_files:
		for file in os.listdir("../results"):
			filename = str("../results/") + os.fsdecode(file)
			if(os.path.isdir(filename)):
				continue
			os.remove(filename)

	print("Done")

if __name__ == "__main__":
	main()