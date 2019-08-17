import sys
sys.path.append(sys.path[0] +"/..")
sys.path.append(sys.path[0])
import numpy as np 
import pandas as pd 
import utils.file_wav as file_wav
from utils.data_set_op import Chinese2Pinyin
from utils.dir_op import get_file_name_ls_from_dir,get_child_dir_ls_from_dir,\
    get_file_dict_for_different_type

class DataSetBase():
    def __init__(self, dataset_dir, label_file = None,label_file_type = 'trn'):
        if dataset_dir[-1]!='/':
            dataset_dir = dataset_dir+'/'
        self.dataset_dir = dataset_dir
        self.label_file = label_file
        self.label_file_type= label_file_type
        self.get_child_file_dict(self.dataset_dir)
    def get_child_file_dict(self,root_dir):
        self.file_dic = get_file_dict_for_different_type(root_dir)

    def read_label_file(self):
        label_dic = {}
        if self.label_file:
            self.label_dic = self.parse_label_file()
        else:
            for file_name,file_dir in self.file_dic["file_dic"][self.label_file_type].items():
                with open(file_dir,'r') as f:
                    lines = f.readlines()
                label_dic[file_name] = self.parse_label_files(lines)
            self.label_dic = label_dic
        
    def parse_label_file(self,labelsplit = ' '):
        label_dic = {}
        with open(self.label_file,'r') as f:
            lines = f.readlines()
            for line in lines:
                file_name = line.split(labelsplit)[0]
                label = line[len(file_name):].strip(labelsplit).strip('\n') 
                label_dic[file_name] = {
                    'chinese':label,
                    'pinyin':Chinese2Pinyin(label)
                }
            return label_dic
    #@abstractmethod
    def parse_label_files(self,lines):
        pass

class thchs30(DataSetBase):
    def __init__(self, dataset_dir, label_file = None,label_file_type = 'trn'):
        return super().__init__( dataset_dir, label_file,label_file_type)
    def parse_label_files(self,lines):
        return_dic = {}
        if len(lines) ==1:
            lines[0].strip('.')
            with open(self.dataset_dir+lines[0].strip('.').strip('\n'),'r' ) as f:
                lines = f.readlines()
        return_dic['chinese'] = lines[0]
        return_dic['pinyin'] = lines[1]
        return return_dic
class aishell(DataSetBase):
    def __init__(self, *args, **kwargs):
        return super().__init__(*args, **kwargs)

class ST_CMDS(DataSetBase):
    def __init__(self, *args, **kwargs):
        return super().__init__(*args, **kwargs)
    def parse_label_files(self,lines):
        return_dic = {}
        return_dic['chinese'] = lines[0]
        return_dic['pinyin'] = Chinese2Pinyin(lines[0])


class DatasetPrepare:
    def __init__(self,dataset_name):
        if dataset_name=="thchs30":
            self.dataset_name = "thchs30"
            self.dataset_root_dir = sys.path[0] +"/../data_thchs30"
            self.train_dir = self.dataset_root_dir + "/train"
            self.test_dir = self.dataset_root_dir + "/test"
        elif dataset_name=="aishell":
            self.dataset_name = "aishell"
            self.dataset_root_dir = sys.path[0] +"/../data_aishell/"
        elif dataset_name=="primewords":
            self.dataset_name = "thchs30"
            self.dataset_root_dir = sys.path[0] +"/../thchs30/"
        elif dataset_name=="ST-CMDS":
            self.dataset_name = "ST-CMDS"
            self.dataset_root_dir = sys.path[0] +"/../ST-CMDS-20170001_1-OS/"        
        elif dataset_name=="train_set":
            self.dataset_name = "train_set"
            self.dataset_root_dir = sys.path[0] +"/../train_set"
    
    def get_wav_and_label(self):
        for file in get_file_name_ls_from_dir(self.train_dir):
            pass

if __name__=="__main__":
    #print(Chinese2Pinyin("你好 我是 顾家新"))
    thchs30 = thchs30("dataset/data_aishell/",label_file = "dataset/data_aishell/transcript/aishell_transcript_v0.8.txt")
    thchs30.read_label_file() 
    print(thchs30.label_dic)
