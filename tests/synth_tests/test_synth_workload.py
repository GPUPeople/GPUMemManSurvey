import sys
sys.path.append('../../scripts')

import os
import shutil
import time
from datetime import datetime
from timedprocess import Command
from Helper import generateResultsFromSynthetic
# from Helper import plotMean
import csv
import argparse

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
	if os.name == 'nt': # If on Windows
		build_path = os.path.join("build", "Release")
		sync_build_path = os.path.join("sync_build", "Release")
	else:
		build_path = "build/"
		sync_build_path = "sync_build/"

	parser = argparse.ArgumentParser(description='Test workload for various frameworks')
	parser.add_argument('-t', type=str, help='Specify which frameworks to test, separated by +, e.g. o+s+h+c+f+r+x+b ---> c : cuda | s : scatteralloc | h : halloc | o : ouroboros | f : fdgmalloc | r : register-efficient | x : xmalloc')
	parser.add_argument('-threadrange', type=str, help='Specify number of threads, given as powers of two, e.g. 0-5 -> results in 1-32')
	parser.add_argument('-range', type=str, help='Sepcify Allocation Range, e.g. 4-1024')
	parser.add_argument('-iter', type=int, help='How many iterations?')
	parser.add_argument('-runtest', action='store_true', default=False, help='Run testcases')
	parser.add_argument('-genres', action='store_true', default=False, help='Generate results')
	parser.add_argument('-genplot', action='store_true', default=False, help='Generate results file and plot')
	parser.add_argument('-testwrite', action='store_true', default=False, help='If set tests write performance, not allocation performance')
	parser.add_argument('-timeout', type=int, help='Timeout Value in Seconds, process will be killed after as many seconds')
	parser.add_argument('-plotscale', type=str, help='log/linear')
	parser.add_argument('-filetype', type=str, help='png or pdf')
	parser.add_argument('-allocsize', type=int, help='How large is the manageable memory in GiB?')
	parser.add_argument('-device', type=int, help='Which device to use', default=0)

	args = parser.parse_args()

	executable_extension = ""
	if os.name == 'nt': # If on Windows
		executable_extension = ".exe"
	# Parse approaches
	if(args.t):
		if any("c" in s for s in args.t):
			testcases["CUDA"] = os.path.join(build_path, str("c_synth_test") + executable_extension)
		if any("x" in s for s in args.t):
			testcases["XMalloc"] = os.path.join(sync_build_path, str("x_synth_test") + executable_extension)
		if any("h" in s for s in args.t):
			testcases["Halloc"] = os.path.join(sync_build_path, str("h_synth_test") + executable_extension)
		if any("s" in s for s in args.t):
			testcases["ScatterAlloc"] = os.path.join(sync_build_path, str("s_synth_test") + executable_extension)
		if any("o" in s for s in args.t):
			testcases["Ouroboros-P-S"] = os.path.join(build_path, str("o_synth_test_p") + executable_extension)
			testcases["Ouroboros-P-VA"] = os.path.join(build_path, str("o_synth_test_vap") + executable_extension)
			testcases["Ouroboros-P-VL"] = os.path.join(build_path, str("o_synth_test_vlp") + executable_extension)
			testcases["Ouroboros-C-S"] = os.path.join(build_path, str("o_synth_test_c") + executable_extension)
			testcases["Ouroboros-C-VA"] = os.path.join(build_path, str("o_synth_test_vac") + executable_extension)
			testcases["Ouroboros-C-VL"] = os.path.join(build_path, str("o_synth_test_vlc") + executable_extension)
		if any("f" in s for s in args.t):
			testcases["FDGMalloc"] = os.path.join(sync_build_path, str("f_synth_test") + executable_extension)
		if any("r" in s for s in args.t):
			# testcases["RegEff-A"] = os.path.join(sync_build_path, str("r_synth_test_a") + executable_extension)
			testcases["RegEff-AW"] = os.path.join(sync_build_path, str("r_synth_test_aw") + executable_extension)
			testcases["RegEff-C"] = os.path.join(sync_build_path, str("r_synth_test_c") + executable_extension)
			testcases["RegEff-CF"] = os.path.join(sync_build_path, str("r_synth_test_cf") + executable_extension)
			testcases["RegEff-CM"] = os.path.join(sync_build_path, str("r_synth_test_cm") + executable_extension)
			testcases["RegEff-CFM"] = os.path.join(sync_build_path, str("r_synth_test_cfm") + executable_extension)
		if any("b" in s for s in args.t):
			testcases["Baseline"] = os.path.join(sync_build_path, str("b_synth_test") + executable_extension)

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
		testwritestr = "0"
		namestr = ""
		if args.testwrite:
			testwritestr = "1"
			namestr = "write_"
		for name, executable in testcases.items():
			csv_path = "results/synth_" + namestr + name + "_" + str(smallest_num_threads)+ "-" + str(largest_num_threads) + "_" + str(smallest_allocation_size) + "-" + str(largest_allocation_size) + ".csv"
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
				run_config = str(num_threads) + " " + str(smallest_allocation_size) + " " + str(largest_allocation_size) + " " + str(num_iterations) + " 0 " + csv_path + " " + str(alloc_size) + " " + testwritestr + " " + str(args.device)
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
	if generate_results:
		if not os.path.exists("results/aggregate"):
			os.mkdir("results/aggregate")
		if args.testwrite:
			generateResultsFromSynthetic(testcases, "results", smallest_num_threads, largest_num_threads, smallest_allocation_size, largest_allocation_size, "Num Threads", "synth_write", 2)
		else:
			generateResultsFromSynthetic(testcases, "results", smallest_num_threads, largest_num_threads, smallest_allocation_size, largest_allocation_size, "Num Threads", "synth", 1)

	print("Done")

if __name__ == "__main__":
	main()