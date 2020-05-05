import time
import os
import pandas
from datetime import datetime
import csv
import numpy as np
import matplotlib.pyplot as plt

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

####################################################################################################
####################################################################################################
# Generate new Results
####################################################################################################
####################################################################################################
def generateResultsFromFileAllocation(param1, param2, param3, dimension_name, output_name_short):
    print("Generate Results for identifier " + str(param1) + "_" + str(param2) + "-" + str(param3))
    # Gather results
    result_alloc = list(list())
    result_free = list(list())

    # Go over files, read data and generate new
    written_header_free = False
    written_header_alloc = False
    for file in os.listdir("results/tmp"):
        filename = str("results/tmp/") + os.fsdecode(file)
        if(os.path.isdir(filename)):
            continue
        if str(param1) != filename.split('_')[3] or str(param2) + "-" + str(param3) != filename.split('_')[4].split(".")[0]:
            continue
        print("Processing -> " + str(filename))
        approach_name = filename.split('_')[2]
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
    alloc_name = str("results/tmp/aggregate/") + time_string + str("_") + output_name_short + str("_alloc_") + str(param1) + "_" + str(param2) + "-" + str(param3) + str(".csv")
    with(open(alloc_name, "w")) as f:
        writer = csv.writer(f, delimiter=',')
        for row in result_alloc:
            writer.writerow(row)
    
    print("Generating -> " + time_string + str("_") + output_name_short + str("_free_") + str(param1) + "_" + str(param2) + "-" + str(param3) + str(".csv"))
    free_name = str("results/tmp/aggregate/") + time_string + str("_") + output_name_short + str("_free_") + str(param1) + "_" + str(param2) + "-" + str(param3) + str(".csv")
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
def plotMean(results, plotscale, plotrange, xlabel, ylabel, title, filename, variant):
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