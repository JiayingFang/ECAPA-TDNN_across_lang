# Preprocess data
########################
## Created by Jiaying ##
########################
import os
import os.path
import sys
import io
import glob
from tqdm import tqdm
import subprocess
from pydub import AudioSegment

def main():
	dir_path = glob.glob('/home/ubuntu/user_space/ECAPA-TDNN-main/data/CN-Celeb_flac/eval/*/*.flac')
	dir_path.sort()
	print('Converting files from FLAC to WAV')
	for fname in tqdm(dir_path):
		outfile = fname.replace('.flac','.wav')
		out = subprocess.call('ffmpeg -v quiet -y -i %s %s' %(fname,outfile), shell=True)
		if out != 0:
			raise ValueError('Conversion failed %s.'%fname)
	return 0

if __name__ == '__main__':
	status = main()
	sys.exit(status)