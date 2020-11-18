import sys
sys.path.append('scripts')
import os
from timedprocess import Command

def main():
	Command("rm -rf build").run()
	Command("rm -rf sync_build").run()

if __name__ == "__main__":
	main()