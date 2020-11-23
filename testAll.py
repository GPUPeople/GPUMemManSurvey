import sys
sys.path.append('scripts')
import os
from timedprocess import Command
import argparse
import shutil

def main():
	parser = argparse.ArgumentParser(description='Run all testcases')
	parser.add_argument('--mem_size', type=int, help='Size of the manageable memory in GB', default=8)
	args = parser.parse_args()

	current_dir = os.getcwd()

	print("The selected amount of manageable memory is: {} Gb".format(str(args.mem_size)))

	# Which tests to run
	tests = {
		"alloc_tests" : [
			["python test_allocation.py -t o+s+h+c+r+x -num 10000 -range 4-32 -iter 50 -runtest -genres -timeout 60 -allocsize {}".format(str(args.mem_size)), "performance"],
			# ["python test_mixed_allocation.py -t o+s+h+c+r+x -num 10000 -range 4-32 -iter 50 -runtest -genres -timeout 60 -allocsize {}".format(str(args.mem_size)), "mixed_performance"],
			# ["python test_scaling.py -t o+s+h+c+r+x -byterange 4-32 -threadrange 0-10 -iter 50 -runtest -genres -timeout 60 -allocsize {}".format(str(args.mem_size)), "scaling"]
		],
		# "frag_tests"  : [
		# 	["python test_fragmentation.py -t o+s+h+c+r+x -num 10000 -range 4-16 -iter 50 -runtest -genres -timeout 60 -allocsize {}".format(str(args.mem_size)), ""],
		# 	["python test_oom.py -t o+s+h+c+r+x -num 10000 -range 4-16 -runtest -genres -timeout 60 -allocsize {}".format(str(args.mem_size)), ""]
		# ],
		# "graph_tests" : [
		# 	["python test_graph_init.py -t o+s+h+c+r+x -configfile config_init.json -runtest -genres -timeout 120", ""],
		# 	["python test_graph_update.py -t o+s+h+c+r+x -configfile config_update.json -runtest -genres -timeout 120", ""],
		# 	["python test_graph_update.py -t o+s+h+c+r+x -configfile config_update_range.json -runtest -genres -timeout 120", ""]
		# ],
		# "synth_tests" : [
		# 	["python test_registers.py -t o+s+h+c+r+x -runtest -genres -allocsize {}".format(str(args.mem_size)), ""],
		# 	["python test_synth_init.py -t o+s+h+c+r+x -runtest -genres -allocsize {}".format(str(args.mem_size)), ""],
		# 	["python test_synth_workload.py -t o+s+h+c+r+x -threadrange 0-10 -range 4-32 -iter 50 -runtest -genres -timeout 60 -allocsize {}".format(str(args.mem_size)), ""],
		# 	["python test_synth_workload.py -t o+s+h+c+r+x -threadrange 0-10 -range 4-32 -iter 5 -runtest -genres -testwrite -timeout 60 -allocsize {}".format(str(args.mem_size)), ""]
		# ]
	}

	for path, commands in tests.items():
		for command in commands:
			print("Will execute command in folder {0} (subfolder {2}): {1}".format(path, command[0], command[1]))

	# Run tests	
	for path, commands in tests.items():
		for command in commands:
			os.chdir(os.path.join(current_dir, "tests", path))
			Command(command[0]).run(timeout=3600)
			shutil.move("tests/{0}/results/{1}aggregate/*".format(path, command[1] + "/"), "results")

if __name__ == "__main__":
	main()