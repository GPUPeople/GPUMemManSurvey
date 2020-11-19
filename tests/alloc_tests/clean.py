import sys
sys.path.append('../../scripts')
import os
from timedprocess import Command
import shutil

def main():
	if os.path.exists("build"):
		shutil.rmtree("build")
	if os.path.exists("sync_build"):
		shutil.rmtree("sync_build")
	print("Cleanup done!")


if __name__ == "__main__":
	main()