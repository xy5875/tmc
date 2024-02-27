import filecmp
import os

folder_a = '/home/dell/xy/AFLvsGFL/data/0/0'
folder_b = '/home/dell/xy/AFLvsGFL/data/1/0'

# 获取两个文件夹中的所有文件列表
files_a = [f for f in os.listdir(folder_a) if os.path.isfile(os.path.join(folder_a, f))]
files_b = [f for f in os.listdir(folder_b) if os.path.isfile(os.path.join(folder_b, f))]

# 比较两个文件夹中的文件是否相同
# common_files = filecmp.dircmp(folder_a, folder_b).common_files



m = 0
for file_aa in files_a:
    for file_bb in files_b:
        path_a = os.path.join(folder_a, file_aa)
        path_b = os.path.join(folder_b, file_bb)
        with open(path_a, 'rb') as file_a, open(path_b, 'rb') as file_b:
            content_a = file_a.read()
            content_b = file_b.read()
        if  content_a == content_b:
            m = m + 1
print("m is ",m)            

# # 检查共同文件是否相同
# for file in common_files:
#     path_a = os.path.join(folder_a, file)
#     path_b = os.path.join(folder_b, file)

#     # 使用适当的方法比较图像，这里假设是通过文件内容比较
#     with open(path_a, 'rb') as file_a, open(path_b, 'rb') as file_b:
#         content_a = file_a.read()
#         content_b = file_b.read()

#     if content_a == content_b:
#         print(f"File '{file}' is identical in both folders.")
#     else:
#         #print(f"File '{file}' is different in both folders.")
#         t = 0
