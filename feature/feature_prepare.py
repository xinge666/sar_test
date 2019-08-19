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
        self.list_symbol = self.GetSymbolList()


    def GetSymbolList(self):
		'''
		加载拼音符号列表，用于标记符号
		返回一个列表list类型变量
		'''
		txt_obj=open('dict.txt','r',encoding='UTF-8') # 打开文件并读入
		txt_text=txt_obj.read()
		txt_lines=txt_text.split('\n') # 文本分割
		list_symbol=[] # 初始化符号列表
		for i in txt_lines:
			if(i!=''):
				txt_l=i.split('\t')
				list_symbol.append(txt_l[0])
		txt_obj.close()
		list_symbol.append('_')
		self.SymbolNum = len(list_symbol)
		return list_symbol

    def SymbolToNum(self,symbol):
		'''
		符号转为数字
		'''
		if(symbol != ''):
			return self.list_symbol.index(symbol)
		return self.SymbolNum

    def get_child_file_dict(self,root_dir):
        """
        获取根目录下所有文件并根据文件类型,使用类型分类
        self.file_dic{
            "file_dic":{file_type1:{file_name1:file_dir1,...},...},
            "train":[file_name1,file_name2,...],
            "dev":[file_name1,file_name2,...],
            "test":[file_name1,file_name2,...],
            "undivided":[file_name1,file_name2,...],
            "file_types":[file_type1,file_type2,...]
        }
        """
        self.file_dic = get_file_dict_for_different_type(root_dir)

    def read_label_file(self):
        """
        解析label文件，获取label_dic
        self.label_dic{
            file_name1:{
                "chinese":XXX,
                "pinyin":XXX,
                "code":XXX,
            },...
        }
        """
        label_dic = {}
        if self.label_file:
            self.label_dic = self.parse_label_file()
        else:
            for file_name,file_dir in self.file_dic["file_dic"][self.label_file_type].items():
                with open(file_dir,'r',encoding='utf-8') as f:
                    lines = f.readlines()
                label_dic[file_name] = self.parse_label_files(lines)
            self.label_dic = label_dic
        for file_name in self.label_dic.keys():
            code_ls = []
            for py in self.label_dic[file_name]['pinyin'].split(' '):
                code_ls.append(self.SymbolToNum(py))
            self.label_dic[file_name]["code"] = code_ls
        
    def parse_label_file(self,labelsplit = ' '):
        """
        如果是单个label文件，解析该文件（子类重写该函数）
        """
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
        """
        如果是多个label文件，解析该文件（子类重写该函数）
        """
        pass

class thchs30(DataSetBase):
    def __init__(self, dataset_dir = sys.path[0]+"/../dataset/data_thchs30", label_file =None,label_file_type = 'trn'):
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
    def __init__(self, dataset_dir = sys.path[0]+"/../dataset/aidatatang_200zh", 
            label_file = sys.path[0]+"/../dataset/aidatatang_200zh/transcript/aidatatang_200_zh_transcript.txt",\
            label_file_type = None):
        return super().__init__( dataset_dir, label_file,label_file_type)

class aishell(DataSetBase):
    def __init__(self, dataset_dir = sys.path[0]+"/../dataset/data_aishell", 
            label_file = sys.path[0]+"/../dataset/data_aishell/transcript/aishell_transcript_v0.8.txt",\
            label_file_type = None):
        return super().__init__( dataset_dir, label_file,label_file_type)


class ST_CMDS(DataSetBase):
    def __init__(self, dataset_dir = sys.path[0]+"/../dataset/ST-CMDS-20170001_1-OS"
                 , label_file =None,label_file_type = 'txt'):
        return super().__init__( dataset_dir, label_file,label_file_type)
    def parse_label_files(self,lines):
        return_dic = {}
        return_dic['chinese'] = lines[0]
        return_dic['pinyin'] = Chinese2Pinyin(lines[0])

class primewords(DataSetBase):
    def __init__(self, dataset_dir = sys.path[0]+"/../dataset/primewords_md_2018_set1", \
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
    def __init__(self, dataset_dir = sys.path[0]+"/../dataset/train", \
            label_file = sys.path[0]+"/../dataset/train/TRANS.txt",\
            label_file_type = None):
        return super().__init__( dataset_dir, label_file,label_file_type)
    def parse_label_file(self,labelsplit ='\t' ):
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
    thchs30 = thchs30() 
    MagicData = MagicData()
    primewords = primewords()
    ST_CMDS = ST_CMDS()



    thchs30.read_label_file() 
    MagicData.read_label_file()
    primewords.read_label_file()
    ST_CMDS.read_label_file()
    import pdb
    pdb.set_trace()
    print(thchs30.label_dic)
