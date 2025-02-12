def remove_before_first_space(s:str):
    # 分割字符串，最大分割次数设为1
    parts = s.split(' ', 1)
    # 如果字符串中有空格，则返回第一个空格后的所有内容
    if len(parts) > 1:
        return parts[1].replace("\"","")
    else:
        # 如果没有空格，返回整个字符串
        return s.replace("\"","")