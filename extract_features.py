import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"

import numpy as np
import torch
from natsort import natsorted
from PIL import Image
from torch.autograd import Variable


def load_frame(frame_file):
	data = Image.open(frame_file)
	data = data.resize((340, 256), Image.ANTIALIAS)
	data = np.array(data)
	data = data.astype(float)
	data = (data * 2 / 255) - 1  # 将图像像素值归一化到 [-1, 1] 范围内
	assert(data.max()<=1.0)
	assert(data.min()>=-1.0)
	return data


def load_rgb_batch(frames_dir, rgb_files, frame_indices):
	batch_data = np.zeros(frame_indices.shape + (256,340,3))  # 索引列表的形状+图像的维度
	# 对于每个索引，加载对应的图像帧，并将其存储在 batch_data 中
	for i in range(frame_indices.shape[0]):
		for j in range(frame_indices.shape[1]):
			batch_data[i,j,:,:,:] = load_frame(os.path.join(frames_dir, rgb_files[frame_indices[i][j]]))
	return batch_data


# 数据增强（过采样）
# “10-crop” augment: cropping images into the center, four corners, and their mirrored counterparts
def oversample_data(data):
	data_flip = np.array(data[:,:,:,::-1,:])  # 沿宽度方向翻转数据数组

	# 从原始数据和翻转后的数据中裁剪出五个不同的区域
	data_1 = np.array(data[:, :, :224, :224, :])
	data_2 = np.array(data[:, :, :224, -224:, :])
	data_3 = np.array(data[:, :, 16:240, 58:282, :])
	data_4 = np.array(data[:, :, -224:, :224, :])
	data_5 = np.array(data[:, :, -224:, -224:, :])

	data_f_1 = np.array(data_flip[:, :, :224, :224, :])
	data_f_2 = np.array(data_flip[:, :, :224, -224:, :])
	data_f_3 = np.array(data_flip[:, :, 16:240, 58:282, :])
	data_f_4 = np.array(data_flip[:, :, -224:, :224, :])
	data_f_5 = np.array(data_flip[:, :, -224:, -224:, :])

	return [data_1, data_2, data_3, data_4, data_5,
		data_f_1, data_f_2, data_f_3, data_f_4, data_f_5]


# 提取特征
def run(i3d, frequency, frames_dir, batch_size, sample_mode):
	assert(sample_mode in ['oversample', 'center_crop'])
	print("batchsize", batch_size)
	chunk_size = 16

	# 将一批数据传递给模型并获取特征
	def forward_batch(b_data):
		b_data = b_data.transpose([0, 4, 1, 2, 3])
		b_data = torch.from_numpy(b_data)   # b,c,t,h,w  # 40x3x16x224x224
		with torch.no_grad():
			b_data = Variable(b_data.cuda()).float()
			inp = {'frames': b_data}
			features = i3d(inp)
		return features.cpu().numpy()
	
	# 使用 natsorted 对帧目录中的文件进行自然排序
	rgb_files = natsorted([f for f in os.listdir(frames_dir) 
						if f.endswith('.jpg')])
	frame_cnt = len(rgb_files)  # 视频片段中所有帧的总数
	# breakpoint()

	# Cut frames
	assert(frame_cnt > chunk_size)
	clipped_length = frame_cnt - chunk_size
	clipped_length = (clipped_length // frequency) * frequency  # The start of last chunk
	frame_indices = [] # Frames to chunks 存储帧的索引
	
	# 根据 batch_size 将帧索引分割成多个批次
	for i in range(clipped_length // frequency + 1):
		frame_indices.append([j for j in range(i * frequency, i * frequency + chunk_size)])
	frame_indices = np.array(frame_indices)
	chunk_num = frame_indices.shape[0]
	batch_num = int(np.ceil(chunk_num / batch_size))    # Chunks to batches
	frame_indices = np.array_split(frame_indices, batch_num, axis=0)
	
	# 根据 sample_mode，初始化 full_features 列表
	if sample_mode == 'oversample':
		full_features = [[] for i in range(10)]
	else:
		full_features = [[]]

	# 对于每个批次，加载图像数据，并根据 sample_mode 进行处理
	for batch_id in range(batch_num): 
		batch_data = load_rgb_batch(frames_dir, rgb_files, frame_indices[batch_id])
		# 对每个裁剪区域运行 forward_batch 并收集特征
		if(sample_mode == 'oversample'):
			batch_data_ten_crop = oversample_data(batch_data)
			for i in range(10):
				assert(batch_data_ten_crop[i].shape[-2]==224)
				assert(batch_data_ten_crop[i].shape[-3]==224)
				temp = forward_batch(batch_data_ten_crop[i])
				full_features[i].append(temp)
		# 只对中心裁剪区域运行 forward_batch 并收集特征
		elif(sample_mode == 'center_crop'):
			batch_data = batch_data[:,:,16:240,58:282,:]
			assert(batch_data.shape[-2]==224)
			assert(batch_data.shape[-3]==224)
			temp = forward_batch(batch_data)
			full_features[0].append(temp)
	
	# 将所有批次的特征合并，并进行适当的转置，以得到最终的特征数组
	full_features = [np.concatenate(i, axis=0) for i in full_features]
	full_features = [np.expand_dims(i, axis=0) for i in full_features]
	full_features = np.concatenate(full_features, axis=0)
	full_features = full_features[:,:,:,0,0,0]
	full_features = np.array(full_features).transpose([1,0,2])
	return full_features
