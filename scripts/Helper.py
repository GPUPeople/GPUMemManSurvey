import time
import os
import pandas
from datetime import datetime
import csv
import numpy as np
import matplotlib.pyplot as plt
import math

colours = {
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
        plt.plot(x_values, y_values, marker='', color=colours[labelname], linewidth=1, label=labelname)
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








# width = 0.125
# 	maxTiming = 0.0
# 	placement = (-3, -2, -1, 0, 1, 2)
# 	colormapping = (0, 6, 5, 4, 2, 1)
# 	approachmapping = (1, 0, 3, 4, 2, 5)

# 	labels = []
# 	xticks = []
# 	for i, t in enumerate(data.testcases):
# 		labels.append(t.testcase[:7] + "...")
# 		xticks.append(index[i] + width)
# 	for i in range(len(data.techniques)):
# 		sizes = []
# 		bottom = []
# 		for t in data.testcases:
# 			if normalizeValues:
# 				# Normalize all values by hiSparse
# 				hisparse_value = t.flops()[SpGEMM_variants_indices.get("ac-SpGEMM")]
				
# 				# timing = (t.Value()[i].times / hisparse_value) * 100
# 				timing = (t.flops()[approachmapping[i]] / hisparse_value) * 100

# 				sizes.append(timing)
# 			else:
# 				# Use standard values
# 				# timing = t.Value()[i].times
# 				timing = t.flops()[approachmapping[i]] / 1000000000
# 				sizes.append(timing)

# 			if timing > maxTiming:
# 				maxTiming = timing
		
# 		techniqueLabel = data.techniques[approachmapping[i]]
# 		if log_used:
# 			bars = ax.bar(index + (placement[i] * width) + width, sizes, log=1, width=width,color=colors[colormapping[i]],align='edge', edgecolor = "black", label=techniqueLabel)
# 		else:
# 			bars = ax.bar(index + (placement[i] * width) + width, sizes, width=width,color=colors[colormapping[i]],align='edge', edgecolor = "black", label=techniqueLabel)

# 	if normalizeValues:
# 		ax.set_ylabel('%')
# 	else:
# 		# ax.set_ylabel('ms')
# 		ax.set_ylabel('GFLOPS')
# 	ax.set_xticks(xticks)
# 	ax.set_xticklabels(labels, fontsize=14)
# 	ax.axis('tight')
# 	ax.set_xlim([-0.4, countTestcases + 0.1])

# 	# ax.set_yticks(np.arange(0, maxTiming + 30, 1.0))
# 	lowerbound = 0
# 	if log_used:
# 		lowerbound = 0.1
# 	upperbound = 25
# 	if log_used:
# 		upperbound = math.pow(10,math.ceil(math.log10(upperbound)))
# 	ax.set_ylim([lowerbound, upperbound])
# 	ax.yaxis.grid(True)
# 	[g.set_alpha(0.75) for g in ax.get_ygridlines()]
	
# 	pplt.setp(ax.get_xticklabels(), rotation=75)
# 	ax.legend(bbox_to_anchor=(0.0 ,1, 0.6, .1), loc=2, ncol=2, mode="expand", frameon=True)
# 	ax.spines['right'].set_color('none')
# 	ax.spines['top'].set_color('none')
# 	ax.xaxis.set_ticks_position('bottom')
# 	ax.yaxis.set_ticks_position('left')
# 	ax.spines['bottom'].set_linewidth(0.5)
# 	ax.spines['left'].set_linewidth(0.5)