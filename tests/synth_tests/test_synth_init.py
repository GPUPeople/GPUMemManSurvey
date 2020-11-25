import sys
sys.path.append('../../scripts')

import os
import shutil
import time
from datetime import datetime
from timedprocess import Command
from Helper import generateResultsFromFileInit
# from Helper import plotInit
import csv
import argparse
import numpy as np

def main():
	# Run all files from a directory
	print("##############################################################################")
	print("Callable as: python test_init_init.py")
	print("##############################################################################")
	
	# Specify which test configuration to use
	testcases = {}
	alloc_size = 8
	generate_results = True
	generate_plots = True
	filetype = "pdf"
	time_out_val = 100
	if os.name == 'nt': # If on Windows
		build_path = os.path.join("build", "Release")
		sync_build_path = os.path.join("sync_build", "Release")
	else:
		build_path = "build/"
		sync_build_path = "sync_build/"

	parser = argparse.ArgumentParser(description='Test framework initialization for various frameworks')
	parser.add_argument('-t', type=str, help='Specify which frameworks to test, separated by +, e.g. o+s+h+c+f+r+x+b ---> c : cuda | s : scatteralloc | h : halloc | o : ouroboros | f : fdgmalloc | r : register-efficient | x : xmalloc')
	parser.add_argument('-runtest', action='store_true', default=False, help='Run testcases')
	parser.add_argument('-genres', action='store_true', default=False, help='Generate results')
	parser.add_argument('-genplot', action='store_true', default=False, help='Generate results file and plot')
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
			testcases["CUDA"] = os.path.join(build_path, str("c_init_test") + executable_extension)
		if any("x" in s for s in args.t):
			testcases["XMalloc"] = os.path.join(sync_build_path, str("x_init_test") + executable_extension)
		if any("h" in s for s in args.t):
			testcases["Halloc"] = os.path.join(sync_build_path, str("h_init_test") + executable_extension)
		if any("s" in s for s in args.t):
			testcases["ScatterAlloc"] = os.path.join(sync_build_path, str("s_init_test") + executable_extension)
		if any("o" in s for s in args.t):
			testcases["Ouroboros-P-S"] = os.path.join(build_path, str("o_init_test_p") + executable_extension)
			testcases["Ouroboros-P-VA"] = os.path.join(build_path, str("o_init_test_vap") + executable_extension)
			testcases["Ouroboros-P-VL"] = os.path.join(build_path, str("o_init_test_vlp") + executable_extension)
			testcases["Ouroboros-C-S"] = os.path.join(build_path, str("o_init_test_c") + executable_extension)
			testcases["Ouroboros-C-VA"] = os.path.join(build_path, str("o_init_test_vac") + executable_extension)
			testcases["Ouroboros-C-VL"] = os.path.join(build_path, str("o_init_test_vlc") + executable_extension)
		if any("f" in s for s in args.t):
			testcases["FDGMalloc"] = os.path.join(sync_build_path, str("f_init_test") + executable_extension)
		if any("r" in s for s in args.t):
			# testcases["RegEff-A"] = os.path.join(sync_build_path, str("r_init_test_a") + executable_extension)
			testcases["RegEff-AW"] = os.path.join(sync_build_path, str("r_init_test_aw") + executable_extension)
			testcases["RegEff-C"] = os.path.join(sync_build_path, str("r_init_test_c") + executable_extension)
			testcases["RegEff-CF"] = os.path.join(sync_build_path, str("r_init_test_cf") + executable_extension)
			testcases["RegEff-CM"] = os.path.join(sync_build_path, str("r_init_test_cm") + executable_extension)
			testcases["RegEff-CFM"] = os.path.join(sync_build_path, str("r_init_test_cfm") + executable_extension)
	
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
			csv_path = "results/init_" + name + "_" + str(alloc_size) + ".csv"
			if(os.path.isfile(csv_path)):
				print("This file already exists, do you really want to OVERWRITE?")
				inputfromconsole = input()
				if not (inputfromconsole == "yes" or inputfromconsole == "y"):
					continue
			with open(csv_path, "w", newline='') as csv_file:
				csv_file.write("AllocationSize (in Byte), mean (ms) GPU, mean (ms) CPU\n")

			run_config = str(alloc_size) + " " + csv_path + " " + str(args.device)
			executecommand = "{0} {1}".format(executable, run_config)
			print("#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#")
			print("Running " + name + " with command -> " + executecommand)
			print("#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#")
			print(executecommand)
			_, process_killed = Command(executecommand).run(timeout=time_out_val)
			if process_killed :
				print("We killed the process!")
				with open(csv_path, "a", newline='') as csv_file:
					csv_file.write("0-------------------> Ran longer than " + str(time_out_val))
			else:
				print("Success!")

	####################################################################################################
	####################################################################################################
	# Generate new Results
	####################################################################################################
	####################################################################################################
	if generate_results:
		if not os.path.exists("results/aggregate"):
			os.mkdir("results/aggregate")
		generateResultsFromFileInit("results", alloc_size, "Bytes", 1)

	# ####################################################################################################
	# ####################################################################################################
	# # Generate new plots
	# ####################################################################################################
	# ####################################################################################################
	# if generate_plots:
	# 	result_init = list()
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
	# 		if filename.split("_")[2] != "init" or str(alloc_size) != filename.split('_')[3].split(".")[0]:
	# 			continue
	# 		# We want the one matching our input
	# 		with open(filename) as f:
	# 			reader = csv.reader(f)
	# 			result_init = list(reader)

	# 	####################################################################################################
	# 	# Lineplot
	# 	####################################################################################################
	# 	plotInit(result_init, 
	# 		testcases,
	# 		plotscale,
	# 		2,
	# 		'Approaches', 
	# 		'ms', 
	# 		"Initialization Timing for " + str(alloc_size) + " GiB initial allocation (GPU)", 
	# 		str("results/plots/") + time_string + "_init_gpu." + filetype)

	# 	####################################################################################################
	# 	# Lineplot with range
	# 	####################################################################################################
	# 	plotInit(result_init, 
	# 		testcases,
	# 		plotscale,
	# 		3,
	# 		'Approaches', 
	# 		'ms',
	# 		"Initialization Timing for " + str(alloc_size) + " GiB initial allocation (CPU)", 
	# 		str("results/plots/") + time_string + "_init_cpu." + filetype)

	print("Done")

if __name__ == "__main__":
	main()