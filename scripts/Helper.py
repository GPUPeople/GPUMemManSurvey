import time
import os
import pandas
from datetime import datetime
import csv
import numpy as np
import matplotlib.pyplot as plt
import math

colours = {
	'ActualSize' : 'black',
	'Halloc' : 'orange' , 
    'XMalloc' : 'silver',
	'Ouroboros-P-VA' : 'lightcoral' , 'Ouroboros-P-VL' : 'darkred' , 'Ouroboros-P-S' : 'red' ,
	'Ouroboros-C-VA' : 'deepskyblue' , 'Ouroboros-C-VL' : 'royalblue' , 'Ouroboros-C-S' : 'navy' ,
	'CUDA' : 'green' , 
	'ScatterAlloc' : 'blue' , 
	'FDGMalloc' : 'gold' , 
	'RegEff-A' : 'mediumvioletred' , 'RegEff-AW' : 'orchid',
	'RegEff-C' : 'purple' , 'RegEff-CF' : 'violet' , 'RegEff-CM' : 'indigo' , 'RegEff-CFM' : 'blueviolet'
}

linestyles = {
	'ActualSize' : 'solid',
	'Halloc' : 'solid' , 
    'XMalloc' : 'solid',
	'Ouroboros-P-VA' : 'dotted' , 'Ouroboros-P-VL' : 'dashed' , 'Ouroboros-P-S' : 'solid' ,
	'Ouroboros-C-VA' : 'dotted' , 'Ouroboros-C-VL' : 'dashed' , 'Ouroboros-C-S' : 'solid' ,
	'CUDA' : 'solid' , 
	'ScatterAlloc' : 'solid' , 
	'FDGMalloc' : 'solid' , 
	'RegEff-A' : 'solid' , 'RegEff-AW' : 'dashed',
	'RegEff-C' : 'solid' , 'RegEff-CF' : 'dashed' , 'RegEff-CM' : 'dotted' , 'RegEff-CFM' : 'dashed'
}

lineplot_width = 20
lineplot_height = 10
barplot_width = 15
barplot_height = 10

####################################################################################################
####################################################################################################
# Generate new Results
####################################################################################################
####################################################################################################
def generateResultsFromFileAllocation(folderpath, param1, param2, param3, dimension_name, output_name_short, approach_pos):
	print("Generate Results for identifier " + str(param1) + "_" + str(param2) + "-" + str(param3))
	# Gather results
	result_alloc = list(list())
	result_free = list(list())

	# Go over files, read data and generate new
	written_header_free = False
	written_header_alloc = False
	for file in os.listdir(folderpath):
		filename = folderpath + str("/") + os.fsdecode(file)
		if(os.path.isdir(filename)):
			continue
		if str(param1) != filename.split('_')[approach_pos+1] or str(param2) + "-" + str(param3) != filename.split('_')[approach_pos+2].split(".")[0]:
			continue
		print("Processing -> " + str(filename))
		approach_name = filename.split('_')[approach_pos]
		with open(filename, newline='') as csv_file:
			dataframe = pandas.read_csv(csv_file)
			if "free" in filename:
				if not written_header_free:
					result_free.append(list(dataframe.iloc[:, 0]))
					result_free[-1].insert(0, dimension_name)
					written_header_free = True
				result_free.append(list(dataframe.iloc[:, 1]))
				result_free[-1].insert(0, approach_name + " - mean")
				result_free.append(list(dataframe.iloc[:, 2]))
				result_free[-1].insert(0, approach_name + " - std_dev")
				result_free.append(list(dataframe.iloc[:, 3]))
				result_free[-1].insert(0, approach_name + " - min")
				result_free.append(list(dataframe.iloc[:, 4]))
				result_free[-1].insert(0, approach_name + " - max")
				result_free.append(list(dataframe.iloc[:, 5]))
				result_free[-1].insert(0, approach_name + " - median")
			else:
				if not written_header_alloc:
					result_alloc.append(list(dataframe.iloc[:, 0]))
					result_alloc[-1].insert(0, dimension_name)
					written_header_alloc = True
				result_alloc.append(list(dataframe.iloc[:, 1]))
				result_alloc[-1].insert(0, approach_name + " - mean")
				result_alloc.append(list(dataframe.iloc[:, 2]))
				result_alloc[-1].insert(0, approach_name + " - std_dev")
				result_alloc.append(list(dataframe.iloc[:, 3]))
				result_alloc[-1].insert(0, approach_name + " - min")
				result_alloc.append(list(dataframe.iloc[:, 4]))
				result_alloc[-1].insert(0, approach_name + " - max")
				result_alloc.append(list(dataframe.iloc[:, 5]))
				result_alloc[-1].insert(0, approach_name + " - median")

	# Get Timestring
	now = datetime.now()
	time_string = now.strftime("%Y-%m-%d_%H-%M-%S")

	# Generate output file
	print("------------------")
	print("Generating -> " + time_string + str("_") + output_name_short + str("_alloc_") + str(param1) + "_" + str(param2) + "-" + str(param3) + str(".csv"))
	alloc_name = folderpath + str("/aggregate/") + time_string + str("_") + output_name_short + str("_alloc_") + str(param1) + "_" + str(param2) + "-" + str(param3) + str(".csv")
	with(open(alloc_name, "w")) as f:
		writer = csv.writer(f, delimiter=',')
		for row in result_alloc:
			writer.writerow(row)

	print("Generating -> " + time_string + str("_") + output_name_short + str("_free_") + str(param1) + "_" + str(param2) + "-" + str(param3) + str(".csv"))
	free_name = folderpath + str("/aggregate/")  + time_string + str("_") + output_name_short + str("_free_") + str(param1) + "_" + str(param2) + "-" + str(param3) + str(".csv")
	with(open(free_name, "w")) as f:
		writer = csv.writer(f, delimiter=',')
		for row in result_free:
			writer.writerow(row)
	print("####################")

std_dev_offset = 1
min_offset = 2
max_offset = 3
median_offset = 4

# Plot mean as a line plot with std-dev
def plotMean(results, testcases, plotscale, plotrange, xlabel, ylabel, title, filename, variant):
	plt.figure(figsize=(lineplot_width, lineplot_height))
	x_values = np.asarray([float(i) for i in results[0][1:]])
	for i in range(1, len(results), 5):
		y_values = None
		if variant == "median":
			y_values = np.asarray([float(i) for i in results[i+median_offset][1:]])
		else:
			y_values = np.asarray([float(i) for i in results[i][1:]])
		y_min = None
		y_max = None
		if variant == "stddev":
			y_stddev = np.asarray([float(i) for i in results[i+std_dev_offset][1:]])
			y_min = y_values-y_stddev
			y_max = y_values+y_stddev
		else:
			y_min = np.asarray([float(i) for i in results[i+min_offset][1:]])
			y_max = np.asarray([float(i) for i in results[i+max_offset][1:]])
		labelname = results[i][0].split(" ")[0]
		if labelname not in testcases:
			continue
		print("Generate plot for " + labelname + " with " + variant)
		plt.plot(x_values, y_values, marker='', color=colours[labelname], linewidth=1, label=labelname, linestyle=linestyles[labelname])
		if plotrange:
			plt.fill_between(x_values, y_min, y_max, alpha=0.5, edgecolor=colours[labelname], facecolor=colours[labelname])
	if plotscale == "log":
		plt.yscale("log")
	plt.ylabel(ylabel)
	plt.xlabel(xlabel)
	plt.title(title)
	plt.legend()
	plt.savefig(filename, dpi=600)

	# Clear Figure
	plt.clf()

# Plot results as a bar plot with std-dev
def plotBars(results, testcases, plotscale, plotrange, xlabel, ylabel, title, filename, variant):
	plt.figure(figsize=(barplot_width, barplot_height))
	num_approaches = int(len(results) / 5)
	width = 0.9 / num_approaches
	index = np.arange(len(results[0][1:]))
	placement = []
	alignlabel = ''
	approach_half = int(math.floor(num_approaches/2))
	error_offset = 0
	if num_approaches % 2 == 0:
		placement = [number - approach_half for number in range(0, num_approaches)]
		alignlabel = 'edge'
		error_offset = width / 2
	else:
		placement = [number - approach_half for number in range(0, num_approaches)]
		alignlabel = 'center'
	labels = []
	xticks = []
	for i in range(len(results[0][1:])):
		labels.append(results[0][1+i])
		xticks.append(index[i])
	x_values = np.asarray([str(i) for i in results[0][1:]])
	j = 0
	for i in range(1, len(results), 5):
		y_values = None
		if variant == "median":
			y_values = np.asarray([float(i) for i in results[i+median_offset][1:]])
		else:
			y_values = np.asarray([float(i) for i in results[i][1:]])
		y_min = None
		y_max = None
		if variant == "stddev":
			y_stddev = np.asarray([float(i) for i in results[i+std_dev_offset][1:]])
			y_min = y_values-y_stddev
			y_min = [max(val, 0) for val in y_min]
			y_max = y_values+y_stddev
		else:
			y_min = np.asarray([float(i) for i in results[i+min_offset][1:]])
			y_max = np.asarray([float(i) for i in results[i+max_offset][1:]])
		labelname = results[i][0].split(" ")[0]
		if labelname not in testcases:
			continue
		yerror = np.array([y_min,y_max])
		outputstring = "Generate plot for " + labelname
		if plotrange:
			outputstring += " with " + variant
		print(outputstring)
		plt.bar(index + (placement[j] * width), y_values, width=width, color=colours[labelname], align=alignlabel, edgecolor = "black", label=labelname, tick_label=x_values)
		if plotrange:
			plt.errorbar(index + (placement[j] * width) + error_offset, y_values, yerror, fmt='r^')
		j += 1
	if plotscale == "log":
		plt.yscale("log")
	plt.ylabel(ylabel)
	plt.xlabel(xlabel)
	plt.xticks(xticks)
	plt.tick_params(axis='x', which='major', labelsize=6)
	plt.tick_params(axis='y', which='major', labelsize=12)
	plt.title(title)
	plt.legend()
	plt.savefig(filename, dpi=600)		

	# Clear Figure
	plt.clf()


def generateResultsFromFileFragmentation(folderpath, param1, param2, param3, dimension_name, approach_pos, iter):
	print("Generate Results for identifier " + str(param1) + "_" + str(param2) + "-" + str(param3))
	# Gather results
	result_frag = list(list())

	# Go over files, read data and generate new
	written_header_frag = False
	for file in os.listdir(folderpath):
		filename = folderpath + str("/") + os.fsdecode(file)
		if(os.path.isdir(filename)):
			continue
		if str(param1) != filename.split('_')[approach_pos+1] or str(param2) + "-" + str(param3) != filename.split('_')[approach_pos+2].split(".")[0]:
			continue
		print("Processing -> " + str(filename))
		approach_name = filename.split('_')[approach_pos]
		with open(filename, newline='') as csv_file:
			dataframe = pandas.read_csv(csv_file)
			if not written_header_frag:
				actual_size = [i for i in range(param2, param3 + 4, 4)]
				result_frag.append(list(actual_size))
				result_frag[-1].insert(0, dimension_name)
				actual_size = [i * param1 for i in range(param2, param3 + 4, 4)]
				result_frag.append(list(actual_size))
				result_frag[-1].insert(0, "ActualSize - range")
				result_frag.append(list(actual_size))
				result_frag[-1].insert(0, "ActualSize - static range after " + str(iter) + " iterations")
				written_header_frag = True
			result_frag.append(list(dataframe.iloc[:, 1]))
			result_frag[-1].insert(0, approach_name + " - range")
			result_frag.append(list(dataframe.iloc[:, iter * 2 - 1]))
			result_frag[-1].insert(0, approach_name + " - static range after " + str(iter) + " iterations")

	# Get Timestring
	now = datetime.now()
	time_string = now.strftime("%Y-%m-%d_%H-%M-%S")

	# Generate output file
	print("------------------")
	print("Generating -> " + time_string + str("_frag_") + str(param1) + "_" + str(param2) + "-" + str(param3) + str(".csv"))
	frag_name = folderpath + str("/aggregate/") + time_string +  str("_frag_") + str(param1) + "_" + str(param2) + "-" + str(param3) + str(".csv")
	with(open(frag_name, "w")) as f:
		writer = csv.writer(f, delimiter=',')
		for row in result_frag:
			writer.writerow(row)
	print("####################")

# Plot mean as a line plot with std-dev
def plotFrag(results, testcases, plotscale, plotrange, xlabel, ylabel, title, filename):
	plt.figure(figsize=(lineplot_width, lineplot_height))
	x_values = np.asarray([float(i) for i in results[0][1:]])
	for i in range(1, len(results)):
		y_values = np.asarray([float(i) for i in results[i][1:]])
		labelname = results[i][0].split(" ")[0]
		print(labelname)
		if labelname not in testcases and labelname != "ActualSize":
			continue
		print("Generate plot for " + labelname)
		plt.plot(x_values, y_values, marker='', color=colours[labelname], linewidth=1, label=labelname, linestyle=linestyles[labelname])
		if plotrange:
			y_max = np.asarray([float(i) for i in results[i+1][1:]])
			plt.fill_between(x_values, y_values, y_max, alpha=0.5, edgecolor=colours[labelname], facecolor=colours[labelname])
	if plotscale == "log":
		plt.yscale("log")
	plt.ylabel(ylabel)
	plt.xlabel(xlabel)
	plt.title(title)
	plt.legend()
	plt.savefig(filename, dpi=600)

	# Clear Figure
	plt.clf()

def generateResultsFromFileOOM(folderpath, param1, param2, param3, dimension_name, approach_pos, alloc_size):
	print("Generate Results for identifier " + str(param1) + "_" + str(param2) + "-" + str(param3))
	# Gather results
	result_oom = list(list())

	# Go over files, read data and generate new
	written_header_frag = False
	for file in os.listdir(folderpath):
		filename = folderpath + str("/") + os.fsdecode(file)
		if(os.path.isdir(filename)):
			continue
		if str(param1) != filename.split('_')[approach_pos+1] or str(param2) + "-" + str(param3) != filename.split('_')[approach_pos+2].split(".")[0]:
			continue
		print("Processing -> " + str(filename))
		approach_name = filename.split('_')[approach_pos]
		with open(filename, newline='') as csv_file:
			csvreader = csv.reader(csv_file, delimiter=',', quotechar='|')
			if not written_header_frag:
				actual_size = [i for i in range(param2, param3 + 4, 4)]
				result_oom.append(list(actual_size))
				result_oom[-1].insert(0, dimension_name)
				actual_size = [(int)(alloc_size / (i * param1)) for i in range(param2, param3 + 4, 4)]
				result_oom.append(list(actual_size))
				result_oom[-1].insert(0, "ActualSize")
				written_header_frag = True
			approach_rounds = [len(row) for row in csvreader]
			approach_rounds = approach_rounds[1:]
			result_oom.append(list(approach_rounds))
			result_oom[-1].insert(0, approach_name)

	# Get Timestring
	now = datetime.now()
	time_string = now.strftime("%Y-%m-%d_%H-%M-%S")

	# Generate output file
	print("------------------")
	print("Generating -> " + time_string + str("_oom_") + str(param1) + "_" + str(param2) + "-" + str(param3) + str(".csv"))
	frag_name = folderpath + str("/aggregate/") + time_string +  str("_oom_") + str(param1) + "_" + str(param2) + "-" + str(param3) + str(".csv")
	with(open(frag_name, "w")) as f:
		writer = csv.writer(f, delimiter=',')
		for row in result_oom:
			writer.writerow(row)
	print("####################")

def generateResultsFromFileInit(folderpath, param1, dimension_name, approach_pos):
	print("Generate Results for identifier " + str(param1))
	# Gather results
	result_init = list(list())

	# Go over files, read data and generate new
	written_header_init = False
	for file in os.listdir(folderpath):
		filename = folderpath + str("/") + os.fsdecode(file)
		if(os.path.isdir(filename)):
			continue
		if str(param1) != filename.split('_')[approach_pos+1].split(".")[0]:
			continue
		print("Processing -> " + str(filename))
		approach_name = filename.split('_')[approach_pos]
		with open(filename, newline='') as csv_file:
			csvreader = csv.reader(csv_file, delimiter=',', quotechar='|')
			if not written_header_init:
				result_init.append(["Approach", "Alloc Size", "Timing (ms) GPU", "Timing (ms) CPU"])
				written_header_init = True
			csvreader = list(csvreader)
			result_init.append(list(csvreader[1]))
			result_init[-1].insert(0, approach_name)

	# Get Timestring
	now = datetime.now()
	time_string = now.strftime("%Y-%m-%d_%H-%M-%S")

	# Generate output file
	print("------------------")
	print("Generating -> " + time_string + str("_init_") + str(param1) + str(".csv"))
	init_name = folderpath + str("/aggregate/") + time_string +  str("_init_") + str(param1) + str(".csv")
	with(open(init_name, "w")) as f:
		writer = csv.writer(f, delimiter=',')
		for row in result_init:
			writer.writerow(row)
	print("####################")

# Plot results as a bar plot with std-dev
def plotInit(results, testcases, plotscale, offset, xlabel, ylabel, title, filename):
	plt.figure(figsize=(barplot_width, barplot_height))
	num_approaches = len(results)
	width = 0.9 / num_approaches
	index = np.arange(len(results[0][1:]))
	placement = []
	alignlabel = ''
	approach_half = int(math.floor(num_approaches/2))
	if num_approaches % 2 == 0:
		placement = [number - approach_half for number in range(0, num_approaches)]
		alignlabel = 'edge'
	else:
		placement = [number - approach_half for number in range(0, num_approaches)]
		alignlabel = 'center'
	labels = []
	xticks = []
	for i in range(len(results[0][1:])):
		labels.append(results[0][1+i])
		xticks.append(index[i])
	# x_values = np.asarray([str(i) for i in results[0][1:]])
	# j = 0
	# for i in range(1, len(results)):
	# 	y_values = None
	# 	if variant == "median":
	# 		y_values = np.asarray([float(i) for i in results[i+median_offset][1:]])
	# 	else:
	# 		y_values = np.asarray([float(i) for i in results[i][1:]])
	# 	y_min = None
	# 	y_max = None
	# 	if variant == "stddev":
	# 		y_stddev = np.asarray([float(i) for i in results[i+std_dev_offset][1:]])
	# 		y_min = y_values-y_stddev
	# 		y_min = [max(val, 0) for val in y_min]
	# 		y_max = y_values+y_stddev
	# 	else:
	# 		y_min = np.asarray([float(i) for i in results[i+min_offset][1:]])
	# 		y_max = np.asarray([float(i) for i in results[i+max_offset][1:]])
	# 	labelname = results[i][0].split(" ")[0]
	# 	if labelname not in testcases:
	# 		continue
	# 	yerror = np.array([y_min,y_max])
	# 	outputstring = "Generate plot for " + labelname
	# 	print(outputstring)
	# 	plt.bar(index + (placement[j] * width), y_values, width=width, color=colours[labelname], align=alignlabel, edgecolor = "black", label=labelname, tick_label=x_values)
	# 	j += 1
	if plotscale == "log":
		plt.yscale("log")
	plt.ylabel(ylabel)
	plt.xlabel(xlabel)
	plt.xticks(xticks)
	plt.tick_params(axis='x', which='major', labelsize=6)
	plt.tick_params(axis='y', which='major', labelsize=12)
	plt.title(title)
	plt.legend()
	plt.savefig(filename, dpi=600)		

	# Clear Figure
	plt.clf()