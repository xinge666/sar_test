import sys
sys.path.append(sys.path[0] +"/..")
sys.path.append(sys.path[0])
import numpy as np 
import pandas as pd 
import utils.file_wav as file_wav
from utils.data_set_op import Chinese2Pinyin
from utils.dir_op import get_file_name_ls_from_dir,get_child_dir_ls_from_dir,\
    get_file_dict_for_different_type
import json


class DataSetBase():
    """
    子类重写 read_label_file label文件解析函数，
    或者
    确定label文件类型后，重写 parse_label_file 解析单个label文件函数 
    """
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
                with open(file_dir,'r',encoding='utf-8') as f:
                    lines = f.readlines()
                label_dic[file_name] = self.parse_label_files(lines)
            self.label_dic = label_dic
        
    def parse_label_file(self,labelsplit = ' '):
        label_dic = {}
        with open(self.label_file,'r',encoding='utf-8') as f:
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
    def __init__(self, dataset_dir = sys.path[0]+"/../data_thchs30", label_file =None,label_file_type = 'trn'):
        return super().__init__( dataset_dir, label_file,label_file_type)
    def parse_label_files(self,lines):
        return_dic = {}
        if len(lines) ==1:
            lines[0].strip('.')
            with open(self.dataset_dir+lines[0].strip('.').strip('\n'),'r' ,encoding='utf-8') as f:
                lines = f.readlines()
        return_dic['chinese'] = lines[0]
        return_dic['pinyin'] = lines[1]
        return return_dic
class aidatatang(DataSetBase):
    def __init__(self, dataset_dir = sys.path[0]+"/../aidatatang_200zh", 
            label_file = sys.path[0]+"/../dataset/aidatatang_200zh/transcript/aidatatang_200_zh_transcript.txt",\
            label_file_type = None):
        return super().__init__( dataset_dir, label_file,label_file_type)

class aishell(DataSetBase):
    def __init__(self, dataset_dir = sys.path[0]+"/../data_aishell", 
            label_file = sys.path[0]+"/../dataset/data_aishell/transcript/aishell_transcript_v0.8.txt",\
            label_file_type = None):
        return super().__init__( dataset_dir, label_file,label_file_type)


class ST_CMDS(DataSetBase):
    def __init__(self, dataset_dir, label_file = None,label_file_type = 'txt'):
        return super().__init__( dataset_dir, label_file,label_file_type)
    def parse_label_files(self,lines):
        return_dic = {}
        return_dic['chinese'] = lines[0]
        return_dic['pinyin'] = Chinese2Pinyin(lines[0])

class primewords(DataSetBase):
    def __init__(self, dataset_dir = sys.path[0]+"/../primewords_md_2018_set1", \
            label_file = sys.path[0]+"/../dataset/primewords_md_2018_set1/set1_transcript.json",\
            label_file_type = None):
        return super().__init__( dataset_dir, label_file,label_file_type)
    def parse_label_file(self,labelsplit = ' '):
        label_dic = {}
        with open(self.label_file,'r',encoding='utf-8') as f:
            lines = f.readlines()
            label_ls = json.loads(lines[0])
            for signle_dic in  label_ls:
                label_name = signle_dic["file"].split('.')[0]
                label_dic[label_name] = {
                    'chinese':signle_dic["text"],
                    'pinyin':Chinese2Pinyin(signle_dic["text"])
                }
            return label_dic

class MagicData(DataSetBase):
    def __init__(self, dataset_dir = sys.path[0]+"/../train", \
            label_file = sys.path[0]+"/../dataset/train/TRANS.txt",\
            label_file_type = None):
        return super().__init__( dataset_dir, label_file,label_file_type)
    def parse_label_file(self,labelsplit = ' '):
        label_dic = {}
        with open(self.label_file,'r',encoding='utf-8') as f:
            lines = f.readlines()
            for line in lines:
                line_ls = line.split(labelsplit)
                file_name = line_ls[0].split('.')[0]
                label = line_ls[-1].strip('\n') 
                label_dic[file_name] = {
                    'chinese':label,
                    'pinyin':Chinese2Pinyin(label)
                }
            return label_dic


if __name__=="__main__":
    #print(Chinese2Pinyin("你好 我是 顾家新"))
    thchs30 = thchs30("dataset/data_aishell/",label_file = "dataset/data_aishell/transcript/aishell_transcript_v0.8.txt")
    thchs30.read_label_file() 
    print(thchs30.label_dic)
