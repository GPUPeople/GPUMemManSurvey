import sys
sys.path.append('scripts')
import os
from timedprocess import Command
import argparse
import time

def main():
	start = time.time()
	parser = argparse.ArgumentParser(description='Setup Project')
	parser.add_argument('--cc', type=int, help='Compute Capability (e.g. 61, 70, 75', default=70)
	args = parser.parse_args()

	current_dir = os.getcwd()

	# Build general testcase
	Command("python clean.py".format()).run()
	Command("python setup.py --cc {}".format(str(args.cc))).run(timeout=3600)

	# Build allocation testcases
	os.chdir(os.path.join(current_dir, "tests/alloc_tests"))
	Command("python clean.py".format()).run()
	Command("python setup.py --cc {}".format(str(args.cc))).run(timeout=3600)

	# Build fragmentation testcases
	os.chdir(os.path.join(current_dir, "tests/frag_tests"))
	Command("python clean.py".format()).run()
	Command("python setup.py --cc {}".format(str(args.cc))).run(timeout=3600)

	# Build graph testcases
	os.chdir(os.path.join(current_dir, "tests/graph_tests"))
	Command("python clean.py".format()).run()
	Command("python setup.py --cc {}".format(str(args.cc))).run(timeout=3600)

	# Build synthetic testcases
	os.chdir(os.path.join(current_dir, "tests/synth_tests"))
	Command("python clean.py".format()).run()
	Command("python setup.py --cc {}".format(str(args.cc))).run(timeout=3600)

	end = time.time()
	timing = end - start
	print("Script finished in {0:.0f} min {1:.0f} sec".format(timing / 60, timing % 60))
	print("Setup done!")

if __name__ == "__main__":
	main()