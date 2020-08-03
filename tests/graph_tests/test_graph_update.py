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

graphs = [
	"144.mtx",  
	"333SP.mtx",
	"adaptive.mtx",
	"caidaRouterLevel.mtx",
	"coAuthorsCiteseer.mtx",
	"delaunay_n20.mtx",
	"fe_body.mtx",
	"hugetric-00000.mtx",
	"in2010.mtx",
	"luxembourg_osm.mtx",
	"rgg_n_2_20_s0.mtx",
	"sc2010.mtx",
	"vsp_mod2_pgp2_slptsk.mtx"
]

# graphs = [
# 	"333SP.mtx",
# 	"adaptive.mtx",
# 	"hugetric-00000.mtx"
# ]

path = "../../../graphs/"

def main():
	# Run all files from a directory
	print("##############################################################################")
	print("Callable as: python test_graph_update.py -h")
	print("##############################################################################")

	config_file = "config_update.json"
	testcases = {}
	build_path = "build/"
	sync_build_path = "sync_build/"
	filetype = "pdf"
	time_out_val = 100
	generate_results = True
	generate_plots = True

	parser = argparse.ArgumentParser(description='Test fragmentation for various frameworks')
	parser.add_argument('-t', type=str, help='Specify which frameworks to test, separated by +, e.g. o+s+h+c+f+r+x ---> c : cuda | s : scatteralloc | h : halloc | o : ouroboros | f : fdgmalloc | r : register-efficient | x : xmalloc')
	parser.add_argument('-configfile', type=str, help='Specify the config file: config.json')
	parser.add_argument('-graphstats', action='store_true', default=False, help='Just write graph stats and do not run testcases')
	parser.add_argument('-runtest', action='store_true', default=False, help='Run testcases')
	parser.add_argument('-genres', action='store_true', default=False, help='Generate results')
	parser.add_argument('-genplot', action='store_true', default=False, help='Generate results file and plot')
	parser.add_argument('-timeout', type=int, help='Timeout Value in Seconds, process will be killed after as many seconds')
	parser.add_argument('-plotscale', type=str, help='log/linear')
	parser.add_argument('-filetype', type=str, help='png or pdf')

	args = parser.parse_args()

	# Parse approaches
	if(args.t):
		if any("c" in s for s in args.t):
			testcases["CUDA"] = build_path + str("c_graph_test")
		if any("x" in s for s in args.t):
			testcases["XMalloc"] = sync_build_path + str("x_graph_test")
		if any("h" in s for s in args.t):
			testcases["Halloc"] = sync_build_path + str("h_graph_test")
		if any("s" in s for s in args.t):
			testcases["ScatterAlloc"] = sync_build_path + str("s_graph_test")
		if any("o" in s for s in args.t):
			# testcases["Ouroboros-P-S"] = build_path + str("o_graph_test_p")
			# testcases["Ouroboros-P-VA"] = build_path + str("o_graph_test_vap")
			# testcases["Ouroboros-P-VL"] = build_path + str("o_graph_test_vlp")
			# testcases["Ouroboros-C-S"] = build_path + str("o_graph_test_c")
			# testcases["Ouroboros-C-VA"] = build_path + str("o_graph_test_vac")
			testcases["Ouroboros-C-VL"] = build_path + str("o_graph_test_vlc")
		if any("f" in s for s in args.t):
			testcases["FDGMalloc"] = sync_build_path + str("f_graph_test")
		if any("r" in s for s in args.t):
			testcases["RegEff-A"] = sync_build_path + str("r_graph_test_a")
			# testcases["RegEff-AW"] = sync_build_path + str("r_graph_test_aw")
			# testcases["RegEff-C"] = sync_build_path + str("r_graph_test_c")
			# testcases["RegEff-CF"] = sync_build_path + str("r_graph_test_cf")
			# testcases["RegEff-CM"] = sync_build_path + str("r_graph_test_cm")
			# testcases["RegEff-CFM"] = sync_build_path + str("r_graph_test_cfm")
	
	# Run Testcases
	run_testcases = args.runtest
	
	# Generate results
	generate_results = args.genres

	# Generate plots
	generate_plots = args.genplot

	# Plot Axis scaling
	plotscale = args.plotscale

	# Config File
	if(args.configfile):
		config_file = args.configfile

	# Timeout (in seconds)
	if(args.timeout):
		time_out_val = args.timeout

	if(args.filetype):
		filetype = args.filetype

	# Sort graphs for consistent ordering
	graphs.sort()
	
	if(args.graphstats):
		csv_path = "results/graph_stats.csv"
		if(os.path.isfile(csv_path)):
			print("This file <" + csv_path + "> already exists, do you really want to OVERWRITE?")
			inputfromconsole = input()
			if not (inputfromconsole == "yes" or inputfromconsole == "y"):
				exit()
		with open(csv_path, "w", newline='') as csv_file:
			csv_file.write("Graph Stats, num vertices, num edges, mean adj length, std-dev, min adj length, max adj length\n")
		for graph in graphs:
			with open(csv_path, "a", newline='') as csv_file:
				csv_file.write(graph + ",")
			run_config = config_file + " " + path + graph + " " + str(1) + " " + csv_path
			executecommand = "{0} {1}".format(build_path + str("c_graph_test"), run_config)
			print("#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#")
			print("Running command -> " + executecommand)
			print("#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#")
			print(executecommand)
			_, process_killed = Command(executecommand).run()
			if process_killed :
				print("We killed the process!")
				with open(csv_path, "a", newline='') as csv_file:
					csv_file.write(",0,0,-------------------> Ran longer than " + str(time_out_val) + "\n")
			else:
				print("Success!")
				with open(csv_path, "a", newline='') as csv_file:
					csv_file.write("\n")
		exit()

	####################################################################################################
	####################################################################################################
	# Run testcases
	####################################################################################################
	####################################################################################################
	if run_testcases:
		for name, executable in testcases.items():
			suffix = ".csv"
			if "range" in config_file:
				suffix = "_range.csv"
			csv_path_insert = "results/graph_update_" + name + "_insert" + suffix
			csv_path_delete = "results/graph_update_" + name + "_delete" + suffix
			if(os.path.isfile(csv_path_insert)):
				print("This file <" + csv_path_insert + "> already exists, do you really want to OVERWRITE?")
				inputfromconsole = input()
				if not (inputfromconsole == "yes" or inputfromconsole == "y"):
					continue
			with open(csv_path_insert, "w", newline='') as csv_file:
				csv_file.write("Graph Update Testcase - " + name + ", mean(ms), std-dev(ms), min(ms), max(ms), median(ms), num iterations\n")
			with open(csv_path_delete, "w", newline='') as csv_file:
				csv_file.write("Graph Update Testcase - " + name + ", mean(ms), std-dev(ms), min(ms), max(ms), median(ms), num iterations\n")
			for graph in graphs:
				with open(csv_path_insert, "a", newline='') as csv_file:
					csv_file.write(graph + ",")
				with open(csv_path_delete, "a", newline='') as csv_file:
					csv_file.write(graph + ",")
				run_config = config_file + " " + path + graph + " " + str(0) + " init.csv " + csv_path_insert + " " + csv_path_delete
				executecommand = "{0} {1}".format(executable, run_config)
				print("#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#")
				print("Running " + name + " with command -> " + executecommand)
				print("#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#")
				print(executecommand)
				_, process_killed = Command(executecommand).run(timeout=time_out_val)
				if process_killed :
					print("We killed the process!")
					with open(csv_path_insert, "a", newline='') as csv_file:
						csv_file.write(",0,0,0,0,0,0,-------------------> Ran longer than " + str(time_out_val) + "\n")
					with open(csv_path_delete, "a", newline='') as csv_file:
						csv_file.write(",0,0,0,0,0,0,-------------------> Ran longer than " + str(time_out_val) + "\n")
				else:
					print("Success!")
					with open(csv_path_insert, "a", newline='') as csv_file:
						csv_file.write("\n")
					with open(csv_path_delete, "a", newline='') as csv_file:
						csv_file.write("\n")


if __name__ == "__main__":
	main()