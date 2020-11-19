import sys
sys.path.append('../../scripts')
import os
from timedprocess import Command
import argparse

def main():
	parser = argparse.ArgumentParser(description='Setup Project')
	parser.add_argument('--cc', type=int, help='Compute Capability', default=70)
	args = parser.parse_args()

	async_flag = "CC{}".format(str(args.cc))
	sync_flag = "CC{}".format(str(args.cc))
	if args.cc >= 70:
		async_flag += "_ASYNC"
		sync_flag += "_SYNC"

	if os.name == 'nt': # If on Windows
		Command("cmake -B build -D{}=ON".format(async_flag)).run()
		Command("msbuild build/GPUMemoryManagers.sln /p:Configuration=Release").run()
		Command("cmake -B sync_build -D{}=ON".format(sync_flag)).run()
		Command("msbuild sync_build/GPUMemoryManagers.sln /p:Configuration=Release").run()
	else: # If on Linux
		Command("mkdir build && cd build && cmake .. -DCMAKE_BUILD_TYPE=Release -D{}=ON".format(async_flag)).run()
		Command("cd build && make").run()
		Command("mkdir sync_build && cd sync_build && cmake .. -DCMAKE_BUILD_TYPE=Release -D{}=ON".format(sync_flag)).run()
		Command("cd sync_build && make").run()

if __name__ == "__main__":
	main()