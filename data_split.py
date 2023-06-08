import jsonlines
import random

# input_files = ['./data/training_data/Task_1_train.jsonl','./data/training_data/Task_1_dev.jsonl']
#
# output_files = ['./newdata/train/Task_1_train.jsonl', \
#                 './newdata/val/Task_1_val.jsonl',\
#                 './newdata/test/Task_1_test.jsonl']
input_files = ['./data/training_data/Task_2_train.jsonl','./data/training_data/Task_2_dev.jsonl']

output_files = ['./newdata/train/Task_2_train.jsonl', \
                './newdata/val/Task_2_val.jsonl',\
                './newdata/test/Task_2_test.jsonl']
# ratios = [0.7, 0.1, 0.2]

ratios=[7,1,2]
# 打开输入文件
with jsonlines.open(input_files[0]) as file1, \
        jsonlines.open(input_files[1]) as file2:
    # 读取所有行
    lines = []
    lines.extend(file1)
    lines.extend(file2)

    # 计算行数比例对应的索引
    total_lines = len(lines)
    ratio_indexes = [0] + [sum(ratios[:i + 1]) for i in range(len(ratios))]
    print(ratio_indexes)

    # 分割并写入输出文件
    for i in range(len(output_files)):
        start_index = int(total_lines * ratio_indexes[i] / sum(ratios))
        end_index = int(total_lines * ratio_indexes[i + 1] / sum(ratios))
        print(start_index, end_index)
        output_data = lines[start_index:end_index]
    #
        with jsonlines.open(output_files[i], mode='w') as output_file:
            output_file.write_all(output_data)
