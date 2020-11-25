import sys
sys.path.append('scripts')
import os
from timedprocess import Command
import argparse
import shutil
import time

def main():
	start = time.time()
	parser = argparse.ArgumentParser(description='Run all testcases')
	parser.add_argument('-mem_size', type=int, help='Size of the manageable memory in GB', default=8)
	parser.add_argument('-device', type=int, help='Which device to use', default=0)
	parser.add_argument('-runtest', action='store_true', default=False, help='Run testcases')
	parser.add_argument('-genres', action='store_true', default=False, help='Generate results')
	args = parser.parse_args()

	current_dir = os.getcwd()

	print("The selected amount of manageable memory is: {0} Gb".format(str(args.mem_size)))

	runteststr = ""
	if args.runtest:
		runteststr = "-runtest"
	genresstr = ""
	if args.genres:
		genresstr = "-genres"

	# Which tests to run
	tests = {
		# "alloc_tests" : [
		# 	["python test_allocation.py -t o+s+h+c+r+x -num 10000 -range 4-32 -iter 50 {0} {1} -timeout 60 -allocsize {2} -device {3}".format(runteststr, genresstr, str(args.mem_size), str(args.device)), "performance"],
		# 	["python test_mixed_allocation.py -t o+s+h+c+r+x -num 10000 -range 4-32 -iter 50 {0} {1} -timeout 30 -allocsize {2} -device {3}".format(runteststr, genresstr, str(args.mem_size), str(args.device)), "mixed_performance"],
		# 	["python test_scaling.py -t o+s+h+c+r+x -byterange 16-32 -threadrange 0-10 -iter 50 {0} {1} -timeout 60 -allocsize {2} -device {3}".format(runteststr, genresstr, str(args.mem_size), str(args.device)), "scaling"]
		# ],
		# "frag_tests"  : [
		# 	["python test_fragmentation.py -t o+s+h+c+r+x -num 10000 -range 4-32 -iter 50 {0} {1} -timeout 60 -allocsize {2} -device {3}".format(runteststr, genresstr, str(args.mem_size), str(args.device)), ""],
		# 	["python test_oom.py -t o+s+h+c+r+x -num 10000 -range 512-512 {0} {1} -timeout 60 -allocsize 2".format(runteststr, genresstr), ""]
		# ],
		# "graph_tests" : [
		# 	["python test_graph_init.py -t o+s+h+c+r+x -configfile config_init.json {0} {1} -timeout 15 -allocsize {2} -device {3} -graphstats".format(runteststr, genresstr, str(args.mem_size), str(args.device)), ""],
		# 	["python test_graph_init.py -t o+s+h+c+r+x -configfile config_init.json {0} {1} -timeout 10 -allocsize {2} -device {3}".format(runteststr, genresstr, str(args.mem_size), str(args.device)), ""],
		# 	["python test_graph_update.py -t o+s+h+c+r+x -configfile config_update.json {0} {1} -timeout 10 -allocsize {2} -device {3}".format(runteststr, genresstr, str(args.mem_size), str(args.device)), ""],
		# 	["python test_graph_update.py -t o+s+h+c+r+x -configfile config_update_range.json {0} {1} -timeout 10 -allocsize {2} -device {3}".format(runteststr, genresstr, str(args.mem_size), str(args.device)), ""]
		# ],
		"synth_tests" : [
			["python test_registers.py -t o+s+h+c+r+x {0} {1} -allocsize {2} -device {3}".format(runteststr, genresstr, str(args.mem_size), str(args.device)), ""],
			# ["python test_synth_init.py -t o+s+h+c+r+x {0} {1} -allocsize {2} -device {3}".format(runteststr, genresstr, str(args.mem_size), str(args.device)), ""],
			# ["python test_synth_workload.py -t b+o+s+h+c+r+x -threadrange 0-10 -range 16-32 -iter 50 {0} {1} -timeout 10 -allocsize {2} -device {3}".format(runteststr, genresstr, str(args.mem_size), str(args.device)), ""],
			# ["python test_synth_workload.py -t b+o+s+h+c+r+x -threadrange 0-10 -range 16-32 -iter 5 {0} {1} -testwrite -timeout 10 -allocsize {2} -device {3}".format(runteststr, genresstr, str(args.mem_size), str(args.device)), ""]
		]
	}

	for path, commands in tests.items():
		for command in commands:
			print("Will execute command in folder {0} (subfolder {2}): {1}".format(path, command[0], command[1]))

	# Run tests
	for path, commands in tests.items():
		for command in commands:
			command_start = time.time()
			full_path = os.path.join(current_dir, "tests", path)
			os.chdir(full_path)
			if not os.path.exists(os.path.join(full_path, "results")):
				os.mkdir(os.path.join(full_path, "results"))
			if command[1] != "":
				if not os.path.exists(os.path.join(full_path, "results", command[1])):
					os.mkdir(os.path.join(full_path, "results", command[1]))
			Command(command[0]).run(timeout=3600)
			if args.genres:
				aggregate_path = os.path.join(full_path, "results", command[1], "aggregate")
				for file in os.listdir(aggregate_path):
					shutil.move(os.path.join(aggregate_path, file), os.path.join(current_dir, "results", file))
			command_end = time.time()
			command_timing = command_end - command_start
			print("Command finished in {0:.0f} min {1:.0f} sec".format(round(command_timing / 60), round(command_timing % 60)))

	end = time.time()
	timing = end - start
	print("Script finished in {0:.0f} min {1:.0f} sec".format(round(timing / 60), round(timing % 60)))

if __name__ == "__main__":
	main()