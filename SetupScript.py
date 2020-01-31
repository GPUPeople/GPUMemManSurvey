from git
import os

def main():
	print("------------------------")
	print("Setup Halloc")
	print("------------------------")

	if not os.path.isdir(frameworks/halloc/repository"):
		git.Git("frameworks/halloc/repository").clone("https://github.com/canonizer/halloc.git")
	else
		print("Halloc Repository already cloned")


if __name__ == "__main__":
	main()