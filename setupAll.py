import sys
sys.path.append('scripts')
import os
from timedprocess import Command
import argparse

def main():
	parser = argparse.ArgumentParser(description='Setup Project')
	parser.add_argument('--cc', type=int, help='Compute Capability', default=70)
	args = parser.parse_args()

	current_dir = os.getcwd()

	# Build general testcase
	Command("python clean.py".format()).run()
	Command("python setup.py --cc {}".format(str(args.cc))).run()

	# Build allocation testcases
	os.chdir(os.path.join(current_dir, "tests/alloc_tests"))
	Command("python clean.py".format()).run()
	Command("python setup.py --cc {}".format(str(args.cc))).run()

	# Build fragmentation testcases
	os.chdir(os.path.join(current_dir, "tests/frag_tests"))
	Command("python clean.py".format()).run()
	Command("python setup.py --cc {}".format(str(args.cc))).run()

	# Build graph testcases
	os.chdir(os.path.join(current_dir, "tests/graph_tests"))
	Command("python clean.py".format()).run()
	Command("python setup.py --cc {}".format(str(args.cc))).run()

	# Build synthetic testcases
	os.chdir(os.path.join(current_dir, "tests/synth_tests"))
	Command("python clean.py".format()).run()
	Command("python setup.py --cc {}".format(str(args.cc))).run()


	print("Setup done!")

if __name__ == "__main__":
	main()