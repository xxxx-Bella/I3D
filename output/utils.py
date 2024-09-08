import os
import numpy as np
import shutil



def copy_and_rename_npy_files(data_dir, i3d_dir):
    """
    找到 data_dir 下的所有 abnormal 目录中的 .npy 文件，拷贝并重命名到 i3d_dir 下
    """
    for root, dirs, files in os.walk(data_dir):
        for dir_name in dirs:
            if dir_name == 'abnormal':  # 寻找 abnormal 目录
                abnormal_dir = os.path.join(root, dir_name)

                seq_name = os.path.basename(root)  # 获取上一级目录 (序列名 seq3)
                scene_name = os.path.basename(os.path.dirname(root))  # 获取上两级目录 (场景名 Crossroads)
                
                # 遍历 abnormal 目录下的 .npy 文件
                for file_name in os.listdir(abnormal_dir):
                    if file_name.endswith('.npy'):
                        npy_file = os.path.join(abnormal_dir, file_name)

                        # 重命名规则: Crossroads_seq3_06_1_label_1_gt.npy
                        new_file_name = f"{scene_name}_{seq_name}_{file_name.split('.')[0]}_label_1_gt.npy"
                        new_file_path = os.path.join(i3d_dir, new_file_name)
                        
                        # 复制文件并重命名
                        shutil.copy(npy_file, new_file_path)
                        # print(f"Copied and renamed: {npy_file} -> {new_file_path}")  


def copy_all_files(src_dir, dest_dir):
    """
    将 src_dir 中的所有文件拷贝到 dest_dir 中 (顺便count)
    """
    vid_n_count = 0
    vid_a_count = 0
    gt_count = 0

    # 确保目标目录存在
    os.makedirs(dest_dir, exist_ok=True)

    # 遍历源目录中的所有文件
    for root, dirs, files in os.walk(src_dir):
        for file_name in files:
            src_file = os.path.join(root, file_name)
            dest_file = os.path.join(dest_dir, file_name)
            
            # copy
            # shutil.copy(src_file, dest_file)
            # print(f"Copied: {src_file} -> {dest_file}")
            
            # count
            parent_dir = os.path.basename(os.path.dirname(src_file)) # .dirname: src_file完整路径的父目录路径
            if parent_dir != 'not-qualified':
                if "_gt.npy" in file_name:
                    gt_count += 1
                elif "label_0.npy" in file_name:
                    vid_n_count += 1
                elif "label_1.npy" in file_name:
                    vid_a_count += 1
            
    print(f'{vid_a_count+vid_n_count} video ({vid_n_count} normal, {vid_a_count} abnormal), {gt_count} gt')


def get_name_list(root_dir):
    files_list = []
    # 获取当前目录下所有的文件
    for filename in os.listdir(root_dir):
        if filename.endswith('.npy'):
            files_list.append(filename)
    
    # 将列表存储到一个文件中
    dataset = 'DA-new' if 'drone_anomaly_new' in root_dir else 'temp'
    list_i3d_path = f'{dataset}-i3d.list'
    list_gt_path = f'{dataset}-gt.list'
    with open(list_i3d_path, 'w') as f:
        for file_name in files_list:
            if '_gt' not in file_name:
                f.write(file_name + '\n')  # 每个文件名占一行
    with open(list_gt_path, 'w') as f:
        for file_name in files_list:
            if '_gt' in file_name:
                f.write(file_name + '\n')

    print(f'List of i3d.npy files has been written to {list_i3d_path}')
    print(f'List of gt.npy files has been written to {list_gt_path}')


def write_absolute_paths(input_list_file, output_list_file):
    input_dir = os.path.join(os.path.dirname(input_list_file), 'drone_anomaly') 
    # 读取文件名列表
    with open(input_list_file, 'r') as f:
        file_names = f.read().splitlines()
    
    # 获取绝对路径并写入到新的文件中
    with open(output_list_file, 'w') as f:
        for file_name in file_names:
            absolute_path = os.path.abspath(os.path.join(input_dir, file_name))
            f.write(absolute_path + '\n')


# # 拷贝 npy in abnormal, 和重命名
# data_dir = '/home/featurize/work/yuxin/data/drone_anomaly_new_tmp'
# i3d_dir = '/home/featurize/work/yuxin/WVAD/I3D/output/drone_anomaly_new_tmp'
# copy_and_rename_npy_files(data_dir, i3d_dir)

# # 文件拷贝
# src_dir = '/home/featurize/work/yuxin/WVAD/I3D/output/drone_anomaly_new' # _tmp
# dest_dir = '/home/featurize/work/yuxin/WVAD/I3D/output/drone_anomaly_new'
# copy_all_files(src_dir, dest_dir)

# # # get i3d and label list (in ./I3D/)
# root_dir = '/home/featurize/work/yuxin/WVAD/I3D/output/drone_anomaly_new'
# get_name_list(root_dir)

# ./RTFM/list/DA-i3d.list = absolute_path(./I3D/output/DA-new-i3d.list)
# also, DA-i3d-gt.list
type = 'test'  # train, test
input_list_file = f'/home/featurize/work/yuxin/WVAD/I3D/output/DA-{type}.list'
output_list_file = f'/home/featurize/work/yuxin/WVAD/RTFM/list/DA-i3d-{type}.list'
# write_absolute_paths(input_list_file, output_list_file)