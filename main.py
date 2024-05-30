from pathlib import Path
from natsort import natsorted
import shutil
import argparse
import numpy as np
import time
# import ffmpeg
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import torch.optim as optim
# from torch.optim import lr_scheduler
# from torch.autograd import Variable
# import torchvision
from extract_features import run
from resnet import i3_res50
import os


def generate(datasetpath, outputpath, pretrainedpath, frequency, batch_size, sample_mode):
	Path(outputpath).mkdir(parents=True, exist_ok=True)
	# temppath = outputpath+ "/temp/"
	rootdir = Path(datasetpath)
	# videos = [str(f) for f in rootdir.glob('**/*.mp4')]
	videos_dir = [str(f) for f in rootdir.glob('*') if not f.name.endswith('.npy')]  # 6 videos
	videos_dir = natsorted(videos_dir) 
	# breakpoint()

	# setup the model
	i3d = i3_res50(400, pretrainedpath)
	i3d.cuda()
	i3d.train(False)  # Set model to evaluate mode

	for video_dir in videos_dir:
		vid_n = video_dir.split("/")[-1].split(".")[0]  # '01'
		scene = video_dir.split("/")[-4]
		seq = video_dir.split("/")[-3]
		vid_label = 0 if video_dir.split("/")[-2] == 'train' else 1
		vid_name = f'{scene}_{seq}_{vid_n}_label_{vid_label}'
		startime = time.time()
		print(f"Generating for {video_dir}")
		# breakpoint()

		# Path(temppath).mkdir(parents=True, exist_ok=True)
		# ffmpeg.input(video).output('{}%d.jpg'.format(temppath),start_number=0).global_args('-loglevel', 'quiet').run()
		# print("Preprocessing done..")
		# features = run(i3d, frequency, temppath, batch_size, sample_mode)

		# 由于图像已经是单独的文件，不需要使用ffmpeg提取帧，直接运行特征提取
		features = run(i3d, frequency, video_dir, batch_size, sample_mode)
		
		np.save(outputpath + "/" + vid_name, features)
		print("Obtained features of size: ", features.shape)  # (56, 10, 2048)
		# shutil.rmtree(temppath)

		print("done in {0}.".format(time.time() - startime))

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--datasetpath', type=str, default="samplevideos/")  # 包含所有视频文件夹的目录 /home/featurize/work/yuxin/data/drone_anomaly/Bike_Roundabout/sequence1/train
	parser.add_argument('--outputpath', type=str, default="output/drone_anomaly")
	parser.add_argument('--pretrainedpath', type=str, default="pretrained/i3d_r50_kinetics.pth")
	parser.add_argument('--frequency', type=int, default=16)
	parser.add_argument('--batch_size', type=int, default=20)
	parser.add_argument('--sample_mode', type=str, default="oversample")
	args = parser.parse_args()
	generate(args.datasetpath, str(args.outputpath), args.pretrainedpath, args.frequency, args.batch_size, args.sample_mode)    
