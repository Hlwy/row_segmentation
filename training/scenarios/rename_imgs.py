import os
import argparse
import numpy as np

def add_filename_prefixs(_dir, _prefix):
	filenames = os.listdir(_dir)
	os.chdir(_dir)
	for file in filenames:
		newName = str(_prefix) + "_" + str(file)
		os.rename(file, newName)
		# print(file)
		# print(newName)
	# print(filenames)
	print("Finished")


#Run Main
if __name__ == "__main__" :

	# Setup commandline argument(s) structures
	ap = argparse.ArgumentParser(description='Image Segmentation')
	ap.add_argument("--prefix", "-p", type=str, default='test', metavar='NAME', help="Name of video file to parse")
	ap.add_argument("--directory", "-d", type=str, default='test', metavar='FOLDER', help="Name of video file to parse")
	# Store parsed arguments into array of variables
	args = vars(ap.parse_args())

	# Extract stored arguments array into individual variables for later usage in script
	prefix = args["prefix"]
	dir = args["directory"]

	add_filename_prefixs(dir, prefix)
