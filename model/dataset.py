from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision import transforms, utils
import numpy as np
import torch
import wave

class ASRDataset(Dataset):
    def __init__(self, label_dic,file_dic, usage = "train",transform=None):
        self.label_dic = label_dic
        self.file_dic = file_dic
        self.usage = usage
        self.transform = transform
     
    def __len__(self):
        """
        继承 Dataset 类后,必须重写的一个方法
        返回数据集的大小
        :return:
        """
        
        return len(self.file_dic[self.usage])
     
    
    def __getitem__(self, idx):
        """
        继承 Dataset 类后,必须重写的一个方法
        返回第 idx 个图像及相关信息
        :param idx:
        :return:
        """
        feature_length = 2000
        label_length = 40
        file_name = self.file_dic[self.usage][idx]
        wav_dir =  self.file_dic["file_dic"]['wav'][file_name]
        feature = self.load_wav_feature(wav_dir)
        while feature.shape[1] < feature_length:
            feature = np.concatenate((feature, feature), axis=1)
        feature = feature[:,:feature_length]
        feature = np.reshape(feature, [1, feature.shape[0], feature.shape[1]])
        
        feature = torch.Tensor(feature)
        
        #feature = torch.tensor(feature).float().unsqueeze(0)
        label = self.label_dic[file_name]['code']
        while len(label) < label_length:
            label.extend(label)
        label = label[:label_length]
        # label拓展
        
        return feature,label,len(label)
    
    
    def load_wav_feature(self, file_dir):
        wav = wave.open(file_dir,"rb") # 打开一个wav格式的声音文件流
        num_frame = wav.getnframes() # 获取帧数
        num_channel=wav.getnchannels() # 获取声道数
        framerate=wav.getframerate() # 获取帧速率
        num_sample_width=wav.getsampwidth() # 获取实例的比特宽度，即每一帧的字节数
        str_data = wav.readframes(num_frame) # 读取全部的帧
        wav.close() # 关闭流
        wave_data = np.fromstring(str_data, dtype = np.short) # 将声音文件数据转换为数组矩阵形式
        wave_data.shape = -1, num_channel # 按照声道数将数组整形，单声道时候是一列数组，双声道时候是两列的矩阵
        wave_data = wave_data.T # 将矩阵转置
        # wave_data = wave_data 
        # return wave_data, framerate
        
        
        x=np.linspace(0, 400 - 1, 400, dtype = np.int64)
        w = 0.54 - 0.46 * np.cos(2 * np.pi * (x) / (400 - 1) ) # 汉明窗
        time_window = 25 # 单位ms
        window_length = fs / 1000 * time_window # 计算窗长度的公式，目前全部为400固定值

        wav_arr = np.array(wavsignal)
        #wav_length = len(wavsignal[0])
        wav_length = wav_arr.shape[1]

        range0_end = int(len(wavsignal[0])/fs*1000 - time_window) // 10 # 计算循环终止的位置，也就是最终生成的窗数
        data_input = np.zeros((range0_end, 200), dtype = np.float) # 用于存放最终的频率特征数据
        data_line = np.zeros((1, 400), dtype = np.float)

        for i in range(0, range0_end):
            p_start = i * 160
            p_end = p_start + 400
            data_line = wav_arr[0, p_start:p_end]
            data_line = data_line * w # 加窗
            data_line = np.abs(fft(data_line)) / wav_length
            data_input[i]=data_line[0:200] # 设置为400除以2的值（即200）是取一半数据，因为是对称的

        #print(data_input.shape)
        data_input = np.log(data_input + 1)
        return data_input

        
    
ASRDataset_ = ASRDataset(thchs30.label_dic,thchs30.file_dic,"train")


def aligin_collate(batch_size):
    """process variable length labels """
    wave_list = list()
    label_list = list()
    length_list = list()
    for _, (wave, label, length) in enumerate(batch_size):
        wave_list.append(wave)
        label_list.extend(label)
        length_list.append(length)

    stacked_wave = torch.stack(wave_list, dim=0)
    label = torch.IntTensor(np.array(label_list))
    length = torch.IntTensor(np.array(length_list))

    return stacked_wave, label, length

train_loader = torch.utils.data.DataLoader(ASRDataset_,
                                           batch_size=3,
                                           collate_fn=aligin_collate,
                                           shuffle=True,
                                           drop_last=True,
                                           num_workers=8)


