"""
此代码用于将GT标注文件中的数据按照ID列进行排序

"""


# 读取文件内容
with open(r'\20240425_6_05_D3_fps5.txt', 'r') as file:
    lines = file.readlines()

# 按ID排序
sorted_lines = sorted(lines, key=lambda x: int(x.split(',')[1]))

# 将排序后的内容写入新文件
with open(r'\20240425_6_05_D3.txt', 'w') as file:
    file.writelines(sorted_lines)
