import os
import sys
import shutil
import time
from timedprocess import Command

Operationcodes = {
	0 : "OKAY",
	1 : "VERIFY_INIT",
	2 : "VERIFY_INSERT",
	3 : "VERIFY_DELETE"
	4 : "FAILED"
}

def main():
	# Run all files from a directory
	print("##############################################################################")
	print("Callable as: python testsystem.py [executable] [foldername]")
	print("##############################################################################")
	
	testLinux = bool()
	if sys.platform == "win32":
		testLinux = False
	else:
		testLinux = True

	executable ="..\\build\\Release\\main.exe"
	foldername = "..\\data"
	if(len(sys.argv) > 1):
		executable = sys.argv[1]
		if(len(sys.argv) > 2):
			foldername = sys.argv[2]
	runconfigstring = "Called with: " + executable + " for data folder: " + foldername
	print(runconfigstring)

	perf_result = "perf.txt"
	return_result = "res.txt"

	print("##############################################################################")
	print("Start Tests")

	program = str()
	if testLinux:
		program = "./" + executable
	else:
		program = executable

	# Go over this directory
	files_tested = list()
	return_codes = list()
	performance_init = list()
	performance_insert = list()
	performance_delete = list()
	for filename in os.listdir(foldername):
		if filename.endswith(".mtx"):
			print("Test " + filename)
			fullpath = foldername  + "/" + filename
			files_tested.append(filename)
			# Cleanup
			if os.path.isfile(perf_result):
				os.remove(perf_result)
			if os.path.isfile(return_result):
				os.remove(return_result)
			executecommand = "{0} {1}".format(program, fullpath)
			print(executecommand)
			Command(executecommand).run(timeout=10)

			# Gather data
			perf_content = str()
			if os.path.isfile(perf_result):
				with open(perf_result) as perf:
					perf_content = perf.readline()
			else:
				perf_content = "0|0|0"
			
			ret_content = str()
			if os.path.isfile(return_result):
				with open(return_result) as ret:
					ret_content = ret.readline()
			else:
				ret_content = "FAILED"

			return_value = int(ret_content)
			if Operationcodes.get(return_value) == "OKAY":
				# We can store the values
				return_codes.append(return_value)
				split_perf = perf_content.split("|")
				performance_init.append(float(split_perf[0]))
				performance_insert.append(float(split_perf[1]))
				performance_delete.append(float(split_perf[2]))
			else:
				return_codes.append(return_value)
		else:
			continue

	print(files_tested)
	print(return_codes)
	print(performance_init)
	print(performance_insert)
	print(performance_delete)

	# Cleanup
	os.remove(perf_result)
	os.remove(return_result)

	l_time = time.localtime()
	print(l_time)
	time_string = str(l_time.tm_year) + "-" + str(l_time.tm_mon) + "-" + str(l_time.tm_mday) + "---" + str(l_time.tm_hour) + "-" + str(l_time.tm_min) + "-" + str(l_time.tm_sec)

	print(time_string)

	with open("log/log-" + time_string + ".log", "w") as logfile:
		for i in range(len(files_tested)):
			logfile.writelines("####################################################\n")
			logfile.writelines(Operationcodes.get(return_codes[i]) + " : " + files_tested[i] + "\n")

	with open("results/res-" + time_string + ".result", "w") as resultsfile:
		for i in range(len(files_tested)):
			resultsfile.writelines("####################################################\n")
			resultsfile.writelines(files_tested[i] + "\n")
			resultsfile.writelines("Init   : " + "{:.3f}".format(performance_init[i]) + " ms\n")
			resultsfile.writelines("Insert : " + "{:.3f}".format(performance_insert[i]) + " ms\n")
			resultsfile.writelines("Delete : " + "{:.3f}".format(performance_delete[i]) + " ms\n")


if __name__ == "__main__":
	main()