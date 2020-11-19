import sys
sys.path.append('scripts')
import os
from timedprocess import Command

def main():
	current_dir = os.getcwd()

	# Build general testcase
	Command("python clean.py".format()).run()

	# Build allocation testcases
	os.chdir(os.path.join(current_dir, "tests/alloc_tests"))
	Command("python clean.py".format()).run()

	# Build fragmentation testcases
	os.chdir(os.path.join(current_dir, "tests/frag_tests"))
	Command("python clean.py".format()).run()

	# Build graph testcases
	os.chdir(os.path.join(current_dir, "tests/graph_tests"))
	Command("python clean.py".format()).run()

	# Build synthetic testcases
	os.chdir(os.path.join(current_dir, "tests/synth_tests"))
	Command("python clean.py".format()).run()

	print("Setup done!")

if __name__ == "__main__":
	main()