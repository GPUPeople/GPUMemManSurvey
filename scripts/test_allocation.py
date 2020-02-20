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

colours = {'Halloc' : 'Orange' , 'Ouroboros' : 'Red' , 'CUDA' : 'green' , 'ScatterAlloc' : 'blue'}

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

	column_names = list()
	column_names.append("Bytes")
	approach_result_alloc = list(list())
	approach_result_alloc.append(np.arange(smallest_allocation_size, largest_allocation_size, 4).tolist())
	approach_result_alloc[0].insert(0, "Bytes")
	approach_result_free = list(list())
	approach_result_free.append(np.arange(smallest_allocation_size, largest_allocation_size, 4).tolist())
	approach_result_free[0].insert(0, "Bytes")

	# Go over files, read data and generate new 
	for file in os.listdir("../results"):
		filename = str("../results/") + os.fsdecode(file)
		if(os.path.isdir(filename)):
			continue
		with open(filename, newline='') as csv_file:
			dataframe = pandas.read_csv(csv_file)
			if "free" in filename:
				approach_result_free.append(list(dataframe.iloc[:, 1]))
				approach_result_free[-1].insert(0, os.fsdecode(file).split('_')[2])
			else:
				approach_result_alloc.append(list(dataframe.iloc[:, 1]))
				approach_result_alloc[-1].insert(0, os.fsdecode(file).split('_')[2])

	# Cleanup
	for file in os.listdir("../results"):
		filename = str("../results/") + os.fsdecode(file)
		if(os.path.isdir(filename)):
			continue
		os.remove(filename)

	now = datetime.now()
	time_string = now.strftime("%b-%d-%Y_%H-%M-%S")

	# Generate output file
	alloc_name = str("../results/allocation/alloc_") + str(num_allocations) + str(".csv")
	with(open(alloc_name, "w")) as f:
		writer = csv.writer(f, delimiter=',')
		for row in approach_result_alloc:
			writer.writerow(row)
	
	free_name = str("../results/free/free_") + str(num_allocations) + str(".csv")
	with(open(free_name, "w")) as f:
		writer = csv.writer(f, delimiter=',')
		for row in approach_result_free:
			writer.writerow(row)

	# Generate output plots
	# Plot allocation plot
	df = pandas.DataFrame({str(approach_result_alloc[0][0]) : approach_result_alloc[0][1:]})
	for i in range(1, len(approach_result_alloc)):
		df[str(approach_result_alloc[i][0])] = approach_result_alloc[i][1:]
	
	for i in range(1, len(approach_result_alloc)):
		plt.plot(str(approach_result_alloc[0][0]), str(approach_result_alloc[i][0]), data=df, marker='', color=colours[str(approach_result_alloc[i][0])], linewidth=2, label=str(approach_result_alloc[i][0]))
	plt.yscale("log")
	plt.ylabel('ms')
	plt.xlabel('Bytes')
	plt.title("Allocation performance for " + str(num_allocations) + " allocations")
	plt.legend()
	plt.savefig(time_string + "_alloc.pdf", dpi=600)

	# Clear Figure
	plt.clf()

	# Plot free plot
	df = pandas.DataFrame({str(approach_result_free[0][0]) : approach_result_free[0][1:]})
	for i in range(1, len(approach_result_free)):
		df[str(approach_result_free[i][0])] = approach_result_free[i][1:]
	
	for i in range(1, len(approach_result_free)):
		plt.plot(str(approach_result_free[0][0]), str(approach_result_free[i][0]), data=df, marker='', color=colours[str(approach_result_free[i][0])], linewidth=2, label=str(approach_result_free[i][0]))
	plt.yscale("log")
	plt.ylabel('ms')
	plt.xlabel('Bytes')
	plt.title("Free performance for " + str(num_allocations) + " allocations")
	plt.legend()
	plt.savefig(str("../results/free/")time_string + "_free.pdf", dpi=600)

	print("Done")











	exit()

	# Timeout (in seconds)
	time_out_val = 10;

	# Run executables
	for executable in testcases:
		smallest_allocation_size = 4
		while smallest_allocation_size <= largest_allocation_size:
			run_config = str(num_allocations) + " " + str(smallest_allocation_size) + " 0 " + str(free_memory)
			executecommand = "{0} {1}".format(executable, run_config)
			print(executecommand)
			Command(executecommand).run(timeout=time_out_val)
			smallest_allocation_size += 4
	
	# Copy resulting files into separate folder with time stamp

	


		shutil.move(filename, "../results/allocation")


if __name__ == "__main__":
	main()