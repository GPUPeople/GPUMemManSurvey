import sys
sys.path.append('../../scripts')
import os
from timedprocess import Command
import argparse

def main():
	parser = argparse.ArgumentParser(description='Setup Project')
	parser.add_argument('--cc', type=int, help='Compute Capability', default=70)
	args = parser.parse_args()

	Command("mkdir build && cd build && cmake .. -DCMAKE_BUILD_TYPE=Release -DCC{}_ASYNC=ON".format(str(args.cc))).run()
	Command("cd build && make").run()
	Command("mkdir sync_build && cd sync_build && cmake .. -DCMAKE_BUILD_TYPE=Release -DCC{}_SYNC=ON".format(str(args.cc))).run()
	Command("cd sync_build && make").run()

if __name__ == "__main__":
	main()