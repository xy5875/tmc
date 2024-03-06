# 从文本文件中读取预测准确率
with open('/data/xy/TMC/data/LABEL.txt', 'r') as file:
    lines = file.readlines()

# 提取预测标签
predicted_labels = []
for line in lines:
    acc_str = line.split(", acc: ")[1].strip()  # 从字符串中提取列表部分
    acc_list = eval(acc_str)  # 将字符串转换为列表
    predicted_label = acc_list.index(max(acc_list))  # 找到最大准确率的索引
    predicted_labels.append(predicted_label)

# 计算准确率
total_images = len(predicted_labels)
correct_predictions = 0
for i in range(total_images):
    true_label = i // 5000  # 真实标签
    if true_label == predicted_labels[i]:
        correct_predictions += 1

accuracy = correct_predictions / total_images
print("Accuracy:", accuracy)
