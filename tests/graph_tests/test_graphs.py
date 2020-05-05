import sys
sys.path.append('../../scripts')

import os
import shutil
import time
from datetime import datetime
from timedprocess import Command
from Helper import generateResultsFromFileAllocation
from Helper import plotMean
import csv
import argparse

def main():
	# Run all files from a directory
	print("##############################################################################")
	print("Callable as: python test_graphs.py -h")
	print("##############################################################################")
	testcases = list()
	config_file = "config.json"

	print("Not yet implemented")


if __name__ == "__main__":
	main()