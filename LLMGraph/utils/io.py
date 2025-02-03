import json
import os
import torch


def readinfo(data_dir):
    file_type = os.path.basename(data_dir).split('.')[1]
    # assert os.path.exists(data_dir),"no such file path: {}".format(data_dir)
    try:
        if file_type == "pt":
            return torch.load(data_dir)
    except Exception as e:
        data_dir = os.path.join(os.path.dirname(data_dir),f"{os.path.basename(data_dir).split('.')[0]}.json")
    try:
        with open(data_dir,'r',encoding = 'utf-8') as f:
            data_list = json.load(f)
            return data_list
    except Exception as e:
        print(e)
        raise ValueError("file type not supported", data_dir)

def writeinfo(data_dir,info):
    file_type = os.path.basename(data_dir).split('.')[1]
    if file_type == "pt":
        torch.save(info,data_dir)
    elif file_type == "json":
        with open(data_dir,'w',encoding = 'utf-8') as f:
            json.dump(info, f, indent=4,separators=(',', ':'),ensure_ascii=False)
        print("write info to {}".format(data_dir))
    else:
        raise ValueError("file type not supported")