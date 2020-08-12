import sys
sys.path.append('../../scripts')

import os
import shutil
import time
from datetime import datetime
from timedprocess import Command
from Helper import generateResultsFromFileAllocation
from Helper import generateResultsFromFileFragmentation
from Helper import plotMean
import csv
import argparse
import numpy as np

def main():
	# Run all files from a directory
	print("##############################################################################")
	print("Callable as: python test_synth_workload.py")
	print("##############################################################################")
	
	# Specify which test configuration to use
	testcases = {}
	num_allocations = 10000
	smallest_allocation_size = 4
	largest_allocation_size = 1024
	alloc_size = 8
	num_iterations = 1
	free_memory = 1
	generate_results = True
	generate_plots = True
	clean_temporary_files = True
	test_warp_based = False
	filetype = "pdf"
	time_out_val = 100
	build_path = "build/"
	sync_build_path = "sync_build/"

	parser = argparse.ArgumentParser(description='Test fragmentation for various frameworks')
	parser.add_argument('-t', type=str, help='Specify which frameworks to test, separated by +, e.g. o+s+h+c+f+r+x+b ---> c : cuda | s : scatteralloc | h : halloc | o : ouroboros | f : fdgmalloc | r : register-efficient | x : xmalloc')
	parser.add_argument('-threadrange', type=str, help='Specify number of threads, given as powers of two, e.g. 0-5 -> results in 1-32')
	parser.add_argument('-range', type=str, help='Sepcify Allocation Range, e.g. 4-1024')
	parser.add_argument('-iter', type=int, help='How many iterations?')
	parser.add_argument('-runtest', action='store_true', default=False, help='Run testcases')
	parser.add_argument('-genres', action='store_true', default=False, help='Generate results')
	parser.add_argument('-genplot', action='store_true', default=False, help='Generate results file and plot')
	parser.add_argument('-timeout', type=int, help='Timeout Value in Seconds, process will be killed after as many seconds')
	parser.add_argument('-plotscale', type=str, help='log/linear')
	parser.add_argument('-filetype', type=str, help='png or pdf')
	parser.add_argument('-allocsize', type=int, help='How large is the manageable memory in GiB?')

	args = parser.parse_args()

	# Parse approaches
	if(args.t):
		if any("c" in s for s in args.t):
			testcases["CUDA"] = build_path + str("c_synth_test")
		if any("x" in s for s in args.t):
			testcases["XMalloc"] = sync_build_path + str("x_synth_test")
		if any("h" in s for s in args.t):
			testcases["Halloc"] = sync_build_path + str("h_synth_test")
		if any("s" in s for s in args.t):
			testcases["ScatterAlloc"] = sync_build_path + str("s_synth_test")
		if any("o" in s for s in args.t):
			testcases["Ouroboros-P-S"] = build_path + str("o_synth_test_p")
			testcases["Ouroboros-P-VA"] = build_path + str("o_synth_test_vap")
			testcases["Ouroboros-P-VL"] = build_path + str("o_synth_test_vlp")
			testcases["Ouroboros-C-S"] = build_path + str("o_synth_test_c")
			testcases["Ouroboros-C-VA"] = build_path + str("o_synth_test_vac")
			testcases["Ouroboros-C-VL"] = build_path + str("o_synth_test_vlc")
		if any("f" in s for s in args.t):
			testcases["FDGMalloc"] = sync_build_path + str("f_synth_test")
		if any("r" in s for s in args.t):
			# testcases["RegEff-A"] = sync_build_path + str("r_synth_test_a")
			# testcases["RegEff-AW"] = sync_build_path + str("r_synth_test_aw")
			testcases["RegEff-C"] = sync_build_path + str("r_synth_test_c")
			testcases["RegEff-CF"] = sync_build_path + str("r_synth_test_cf")
			# testcases["RegEff-CM"] = sync_build_path + str("r_synth_test_cm")
			# testcases["RegEff-CFM"] = sync_build_path + str("r_synth_test_cfm")
		if any("b" in s for s in args.t):
			testcases["Baseline"] = build_path + str("b_synth_test")
	
	# Parse range
	if(args.threadrange):
		selected_range = args.threadrange.split('-')
		smallest_num_threads = 2 ** int(selected_range[0])
		largest_num_threads = 2 ** int(selected_range[1])

	# Parse range
	if(args.range):
		selected_range = args.range.split('-')
		smallest_allocation_size = int(selected_range[0])
		largest_allocation_size = int(selected_range[1])
	
	# Parse num iterations
	if(args.iter):
		num_iterations = args.iter

	# Run Testcases
	run_testcases = args.runtest
	
	# Generate results
	generate_results = args.genres

	# Generate plots
	generate_plots = args.genplot

	# Plot Axis scaling
	plotscale = args.plotscale

	# Timeout (in seconds)
	if(args.timeout):
		time_out_val = args.timeout

	if(args.filetype):
		filetype = args.filetype

	if(args.allocsize):
		alloc_size = args.allocsize 

	####################################################################################################
	####################################################################################################
	# Run testcases
	####################################################################################################
	####################################################################################################
	if run_testcases:
		for name, executable in testcases.items():
			csv_path = "results/synth_write_" + name + "_" + str(smallest_num_threads)+ "-" + str(largest_num_threads) + "_" + str(smallest_allocation_size) + "-" + str(largest_allocation_size) + ".csv"
			if(os.path.isfile(csv_path)):
				print("This file <" + csv_path + "> already exists, do you really want to OVERWRITE?")
				inputfromconsole = input()
				if not (inputfromconsole == "yes" or inputfromconsole == "y"):
					continue
			with open(csv_path, "w", newline='') as csv_file:
				csv_file.write("NumThreads, mean, std-dev, min, max, median\n")
			num_threads = smallest_num_threads
			while num_threads <= largest_num_threads:
				with open(csv_path, "a", newline='') as csv_file:
					csv_file.write(str(num_threads) + ", ")
				run_config = str(num_threads) + " " + str(smallest_allocation_size) + " " + str(largest_allocation_size) + " " + str(num_iterations) + " 0 " + csv_path + " " + str(alloc_size)
				executecommand = "{0} {1}".format(executable, run_config)
				print("#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#")
				print("Running " + name + " with command -> " + executecommand)
				print("#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#")
				print(executecommand)
				_, process_killed = Command(executecommand).run(timeout=time_out_val)
				if process_killed :
					print("We killed the process!")
					with open(csv_path, "a", newline='') as csv_file:
						csv_file.write("0,0,-------------------> Ran longer than " + str(time_out_val) + "\n")
				else:
					print("Success!")
					with open(csv_path, "a", newline='') as csv_file:
						csv_file.write("\n")
				num_threads *= 2

	# ####################################################################################################
	# ####################################################################################################
	# # Generate new Results
	# ####################################################################################################
	# ####################################################################################################
	# if generate_results:
	# 	generateResultsFromFileFragmentation("results", num_allocations, smallest_allocation_size, largest_allocation_size, "Bytes", 1, num_iterations)

	# ####################################################################################################
	# ####################################################################################################
	# # Generate new plots
	# ####################################################################################################
	# ####################################################################################################
	# if generate_plots:
	# 	result_frag = list()
	# 	# Get Timestring
	# 	now = datetime.now()
	# 	time_string = now.strftime("%b-%d-%Y_%H-%M-%S")

	# 	if plotscale == "log":
	# 		time_string += "_log"
	# 	else:
	# 		time_string += "_lin"

	# 	for file in os.listdir("results/aggregate"):
	# 		filename = str("results/aggregate/") + os.fsdecode(file)
	# 		if(os.path.isdir(filename)):
	# 			continue
	# 		if filename.split("_")[2] != "frag" or str(num_allocations) != filename.split('_')[3] or str(smallest_allocation_size) + "-" + str(largest_allocation_size) != filename.split('_')[4].split(".")[0]:
	# 			continue
	# 		# We want the one matching our input
	# 		with open(filename) as f:
	# 			reader = csv.reader(f)
	# 			result_frag = list(reader)

	# 	####################################################################################################
	# 	# Lineplot
	# 	####################################################################################################
	# 	plotFrag(result_frag, 
	# 		testcases,
	# 		plotscale,
	# 		False, 
	# 		'Bytes', 
	# 		'Byte - Range', 
	# 		"Fragmentation: Byte-Range for " + str(num_allocations) + " allocations", 
	# 		str("results/plots/") + time_string + "_frag." + filetype)

	# 	####################################################################################################
	# 	# Lineplot with range
	# 	####################################################################################################
	# 	plotFrag(result_frag, 
	# 		testcases,
	# 		plotscale,
	# 		True, 
	# 		'Bytes', 
	# 		'Byte - Range',
	# 		"Fragmentation: Byte-Range for " + str(num_allocations) + " allocations", 
	# 		str("results/plots/") + time_string + "_frag_range." + filetype)

	print("Done")

if __name__ == "__main__":
	main()