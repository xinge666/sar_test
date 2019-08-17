import os

def get_file_name_ls_from_dir(file_dir):
    for root, dirs, files in os.walk(file_dir):
        return files

def get_child_dir_ls_from_dir(file_dir):
    for root, dirs, files in os.walk(file_dir):
        return dirs

def get_all_files_name_ls(file_dir,flie_ls = []):
    now_files = [file_dir+"/"+ file for file in get_file_name_ls_from_dir(file_dir) if not file[0] =="." ]
    flie_ls.extend(now_files)
    for dir in  get_child_dir_ls_from_dir(file_dir):
        flie_ls.extend(get_all_files_name_ls(file_dir+"/"+dir,flie_ls))
    return flie_ls

def get_file_dict_for_different_type(file_dir):
    file_ls = get_all_files_name_ls(file_dir)
    file_names =[]
    train_files =[]
    dev_files = []
    test_files =[]
    undivided= []
    file_types = set()
    dic_root = {"file_dic":{}}
    for flie in file_ls:
        file_temp = flie.split('/')[-1]
        file_name = file_temp.split('.')[0]
        file_type = file_temp.split('.')[-1]
        file_types.add(file_type)
        if file_type in dic_root["file_dic"].keys():
            dic_root["file_dic"][file_type][file_name] = flie
        else:
            dic_root["file_dic"][file_type] = {}
            dic_root["file_dic"][file_type][file_name] = flie

        if "train" in flie:
            train_files.append(file_name)
        elif "dev" in flie:
            dev_files.append(file_name)
        elif "test" in flie:
            test_files.append(file_name)
        else:
            undivided.append(file_name)
            
    dic_root['train'] = train_files
    dic_root['dev'] = dev_files
    dic_root['test'] = test_files
    undivided = set(undivided) - set(train_files)- set(dev_files)- set(test_files)
    dic_root['undivided'] = undivided
    dic_root['file_types'] = list(file_types)
    return dic_root
if __name__=="__main__":
    dic_root = get_file_dict_for_different_type("../utils")

