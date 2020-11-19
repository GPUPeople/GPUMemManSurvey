import shutil
import os

def main():
	shutil.copy("frameworks/halloc/fixes/grid.cuh", "frameworks/halloc/repository/src/grid.cuh")
	shutil.copy("frameworks/halloc/fixes/slab.cuh", "frameworks/halloc/repository/src/slab.cuh")
	shutil.copy("frameworks/halloc/fixes/utils.h", "frameworks/halloc/repository/src/utils.h")
	shutil.copy("frameworks/halloc/fixes/globals.cuh", "frameworks/halloc/repository/src/globals.cuh")
	shutil.copy("frameworks/halloc/fixes/size-info.h", "frameworks/halloc/repository/src/size-info.h")
	shutil.copy("frameworks/halloc/fixes/size-info.cuh", "frameworks/halloc/repository/src/size-info.cuh")
	print("Fixed Halloc")
	print("Did you install Boost? (y/n)")
	inputfromconsole = input()
	if not (inputfromconsole == "yes" or inputfromconsole == "y"):
		exit()
	if os.name == 'nt': # If on Windows
		print("Did you set BOOST_DIR in BaseCMake.cmake? (y/n)")
		inputfromconsole = input()
		if not (inputfromconsole == "yes" or inputfromconsole == "y"):
			exit()
	print("Init done!")

if __name__ == "__main__":
	main()