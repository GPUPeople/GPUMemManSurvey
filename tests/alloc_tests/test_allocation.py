import sys
sys.path.append('../../scripts')

import os
import shutil
import time
from datetime import datetime
from timedprocess import Command
from Helper import generateResultsFromFileAllocation
from Helper import plotMean
import csv
import argparse

def main():
	print("##############################################################################")
	print("Callable as: python test_allocation.py -h")
	print("##############################################################################")
	
	# Specify which test configuration to use
	testcases = {}
	num_allocations = 10000
	smallest_allocation_size = 4
	largest_allocation_size = 1024
	num_iterations = 25
	free_memory = 1
	filetype = "pdf"
	time_out_val = 10
	build_path = "build/"
	sync_build_path = "sync_build/"

	parser = argparse.ArgumentParser(description='Test allocation performance for various frameworks')
	parser.add_argument('-t', type=str, help='Specify which frameworks to test, separated by +, e.g. o+s+h+c+f+r+x ---> c : cuda | s : scatteralloc | h : halloc | o : ouroboros | f : fdgmalloc | r : register-efficient | x : xmalloc')
	parser.add_argument('-num', type=int, help='How many allocations to perform')
	parser.add_argument('-range', type=str, help='Specify Allocation Range, e.g. 4-1024')
	parser.add_argument('-iter', type=int, help='How many iterations?')
	parser.add_argument('-runtest', action='store_true', default=False, help='Run testcases')
	parser.add_argument('-genres', action='store_true', default=False, help='Generate results')
	parser.add_argument('-genplot', action='store_true', default=False, help='Generate results file and plot')
	parser.add_argument('-cleantemp', action='store_true', default=False, help='Clean up temporary files')
	parser.add_argument('-warp', action='store_true', default=False, help='Start testcases warp-based')
	parser.add_argument('-devmeasure', action='store_true', default=False, help='Measure performance on device in cycles')
	parser.add_argument('-plotscale', type=str, help='log/linear')
	parser.add_argument('-timeout', type=int, help='Timeout Value in Seconds, process will be killed after as many seconds')
	parser.add_argument('-filetype', type=str, help='png or pdf')

	args = parser.parse_args()

	# Parse approaches
	if(args.t):
		if any("c" in s for s in args.t):
			testcases["CUDA"] = build_path + str("c_alloc_test")
		if any("x" in s for s in args.t):
			testcases["XMalloc"] = sync_build_path + str("x_alloc_test")
		if any("h" in s for s in args.t):
			testcases["Halloc"] = sync_build_path + str("h_alloc_test")
		if any("s" in s for s in args.t):
			testcases["ScatterAlloc"] = sync_build_path + str("s_alloc_test")
		if any("o" in s for s in args.t):
			# testcases["Ouroboros-P-S"] = build_path + str("o_alloc_test_p")
			# testcases["Ouroboros-P-VA"] = build_path + str("o_alloc_test_vap")
			# testcases["Ouroboros-P-VL"] = build_path + str("o_alloc_test_vlp")
			testcases["Ouroboros-C-S"] = build_path + str("o_alloc_test_c")
			# testcases["Ouroboros-C-VA"] = build_path + str("o_alloc_test_vac")
			testcases["Ouroboros-C-VL"] = build_path + str("o_alloc_test_vlc")
		if any("f" in s for s in args.t):
			testcases["FDGMalloc"] = sync_build_path + str("f_alloc_test")
		if any("r" in s for s in args.t):
			testcases["RegEff-A"] = sync_build_path + str("r_alloc_test_a")
			testcases["RegEff-AW"] = sync_build_path + str("r_alloc_test_aw")
			# testcases["RegEff-C"] = sync_build_path + str("r_alloc_test_c")
			# testcases["RegEff-CF"] = sync_build_path + str("r_alloc_test_cf")
			# testcases["RegEff-CM"] = sync_build_path + str("r_alloc_test_cm")
			# testcases["RegEff-CFM"] = sync_build_path + str("r_alloc_test_cfm")
	
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

	# Plot Axis scaling
	plotscale = args.plotscale

	# Clean temporary files
	clean_temporary_files = args.cleantemp

	# Measure on device
	if args.devmeasure:
		measure_on_device = 1
	else:
		measure_on_device = 0

	# Currently we cannot measure on the device when running warp-based
	if measure_on_device and test_warp_based:
		print("Cannot measure on device and warp-based at the same time!")
		exit(-1)

	# Timeout (in seconds)
	if(args.timeout):
		time_out_val = args.timeout
	
	if(args.filetype):
		filetype = args.filetype

	####################################################################################################
	####################################################################################################
	# Run testcases
	####################################################################################################
	####################################################################################################
	if run_testcases:
		# Run Testcase
		for name, path in testcases.items():
			csv_path_alloc = "results/performance/perf_alloc_" + name + "_" + str(num_allocations) + "_" + str(smallest_allocation_size) + "-" + str(largest_allocation_size) + ".csv"
			csv_path_free = "results/performance/perf_free_" + name + "_" + str(num_allocations) + "_" + str(smallest_allocation_size) + "-" + str(largest_allocation_size) + ".csv"
			if(os.path.isfile(csv_path_alloc)):
				print("This file <" + csv_path_alloc + "> already exists, do you really want to OVERWRITE?")
				inputfromconsole = input()
				if not (inputfromconsole == "yes" or inputfromconsole == "y"):
					continue
			with open(csv_path_alloc, "w", newline='') as csv_file:
				csv_file.write("AllocationSize (in Byte), mean, std-dev, min, max, median")
			with open(csv_path_free, "w", newline='') as csv_file:
				csv_file.write("AllocationSize (in Byte), mean, std-dev, min, max, median")
			allocation_size = smallest_allocation_size
			while allocation_size <= largest_allocation_size:
				with open(csv_path_alloc, "a", newline='') as csv_file:
					csv_file.write("\n" + str(allocation_size) + ",")
				with open(csv_path_free, "a", newline='') as csv_file:
					csv_file.write("\n" + str(allocation_size) + ",")
				run_config = str(num_allocations) + " " + str(allocation_size) + " " + str(num_iterations) + " " + str(measure_on_device) + " " + str(test_warp_based) + " 1 " + str(free_memory) + " " + csv_path_alloc + " " + csv_path_free
				executecommand = "{0} {1}".format(path, run_config)
				print("#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#")
				print("Running " + name + " with command -> " + executecommand)
				print("#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#")
				_, process_killed = Command(executecommand).run(timeout=time_out_val)
				if process_killed :
					print("We killed the process!")
					with open(csv_path_alloc, "a", newline='') as csv_file:
						csv_file.write("0.00,0.00,0.00,0.00,0.00,-------------------> Ran longer than " + str(time_out_val))
					with open(csv_path_free, "a", newline='') as csv_file:
						csv_file.write("0.00,0.00,0.00,0.00,0.00,-------------------> Ran longer than " + str(time_out_val))
				else:
					print("Success!")
				allocation_size += 4

	####################################################################################################
	####################################################################################################
	# Generate new Results
	####################################################################################################
	####################################################################################################
	if generate_results:
		generateResultsFromFileAllocation(testcases, "results/performance", num_allocations, smallest_allocation_size, largest_allocation_size, "Bytes", "perf", 2)
	
	####################################################################################################
	####################################################################################################
	# Generate new plots
	####################################################################################################
	####################################################################################################
	if generate_plots:
		result_alloc = list(list())
		result_free = list(list())
		# Get Timestring
		now = datetime.now()
		time_string = now.strftime("%b-%d-%Y_%H-%M-%S")

		if plotscale == "log":
			time_string += "_log"
		else:
			time_string += "_lin"

		for file in os.listdir("results/performance/aggregate"):
			filename = str("results/performance/aggregate/") + os.fsdecode(file)
			if(os.path.isdir(filename)):
				continue
			if filename.split("_")[2] != "perf" or str(num_allocations) != filename.split('_')[4] or str(smallest_allocation_size) + "-" + str(largest_allocation_size) != filename.split('_')[5].split(".")[0]:
				continue
			# We want the one matching our input
			with open(filename) as f:
				reader = csv.reader(f)
				if "free" in filename:
					result_free = list(reader)
				else:
					result_alloc = list(reader)

		####################################################################################################
		# Alloc - Mean - Std-dev
		####################################################################################################
		plotMean(result_alloc, 
			testcases,
			plotscale,
			False, 
			'Bytes', 
			'ms', 
			"Allocation performance for " + str(num_allocations) + " allocations (mean)", 
			str("results/plots/performance/") + time_string + "_alloc." + filetype,
			"stddev")
		print("---------------------------------------")
		plotMean(result_alloc, 
			testcases,
			plotscale,
			True, 
			'Bytes', 
			'ms', 
			"Allocation performance for " + str(num_allocations) + " allocations (mean + std-dev)", 
			str("results/plots/performance/") + time_string + "_alloc_stddev." + filetype,
			"stddev")
		print("---------------------------------------")
		####################################################################################################
		# Free - Mean - Std-dev
		####################################################################################################
		plotMean(result_free, 
			testcases,
			plotscale,
			False,
			'Bytes', 
			'ms', 
			"Free performance for " + str(num_allocations) + " allocations (mean)", 
			str("results/plots/performance/") + time_string + "_free." + filetype,
			"stddev")
		print("---------------------------------------")
		plotMean(result_free, 
			testcases,
			plotscale,
			True,
			'Bytes', 
			'ms', 
			"Free performance for " + str(num_allocations) + " allocations (mean + std-dev)", 
			str("results/plots/performance/") + time_string + "_free_stddev." + filetype,
			"stddev")
		print("---------------------------------------")
		####################################################################################################
		# Alloc - Mean - Min/Max
		####################################################################################################
		plotMean(result_alloc, 
			testcases,
			plotscale,
			True,
			'Bytes', 
			'ms', 
			"Allocation performance for " + str(num_allocations) + " allocations (mean + min/max)", 
			str("results/plots/performance/") + time_string + "_alloc_min_max." + filetype,
			"minmax")
		print("---------------------------------------")
		####################################################################################################
		# Free - Mean - Min/Max
		####################################################################################################
		plotMean(result_free, 
			testcases,
			plotscale,
			True,
			'Bytes', 
			'ms', 
			"Free performance for " + str(num_allocations) + " allocations (mean + min/max)", 
			str("results/plots/performance/") + time_string + "_free_min_max." + filetype,
			"minmax")
		print("---------------------------------------")
		####################################################################################################
		# Alloc - Median
		####################################################################################################
		plotMean(result_alloc, 
			testcases,
			plotscale,
			False,
			'Bytes', 
			'ms', 
			"Allocation performance for " + str(num_allocations) + " allocations (median)", 
			str("results/plots/performance/") + time_string + "_alloc_median." + filetype,
			"median")
		print("---------------------------------------")
		####################################################################################################
		# Free - Median
		####################################################################################################
		plotMean(result_free, 
			testcases,
			plotscale,
			False,
			'Bytes', 
			'ms', 
			"Free performance for " + str(num_allocations) + " allocations (median)", 
			str("results/plots/performance/") + time_string + "_free_median." + filetype,
			"median")
		print("---------------------------------------")

	####################################################################################################
	####################################################################################################
	# Clean temporary files
	####################################################################################################
	####################################################################################################
	if clean_temporary_files:
		print("Do you REALLY want to delete all temporary files?:")
		inputfromconsole = input()
		if not (inputfromconsole == "yes" or inputfromconsole == "y"):
			exit(-1)
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