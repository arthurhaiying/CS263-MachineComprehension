import torch
from transformers import RobertaTokenizer, RobertaForSequenceClassification
from torch.utils.data import Dataset, DataLoader

import json
PLACE_HOLDER = '@placeholder'
def substitute(question, opt):
    question = question.replace(PLACE_HOLDER, opt)
    return question
def json_query_complete(json_instance):

        article = json_instance["article"]
        question = json_instance["question"]
        opt0, opt1, opt2, opt3, opt4 = json_instance["option_0"], json_instance["option_1"], \
                json_instance["option_2"], json_instance["option_3"], json_instance["option_4"]
        opts = [opt0, opt1, opt2, opt3, opt4]
        label = json_instance["label"]
        instance_tuple = (article, question, opts, label)

        return instance_tuple

def transform_binary(instance,new_label, new_query):
    article, question, opts, label = instance

    # scores = [0.0] * len(opts)
    # scores[label] = 1.0
    questions = [substitute(question, opt) for opt in opts]

    for i in range(len(questions)):
        Q_i = questions[i]
        seq_i =Q_i+'</s></s>'+article
        # seq_i = Q_i + '[SEP]' + article
        label_i = 1 if label==i else 0
        new_label.append(label_i)
        new_query.append(seq_i)
class MyDataset(Dataset):
    def __init__(self, file_path, tokenizer, max_len=512):
        self.data = []
        self.label =[]
        self.tokenizer = tokenizer
        self.max_len = max_len

        # 从JSONL文件中读取数据
        with open(file_path, 'r') as file:
            for line in file:
                new_label = []
                new_query =[]
                json_obj = json.loads(line)
                json_instance = json_query_complete(json_obj)
                transform_binary(json_instance, new_label, new_query)
                # print(len(new_query))
                # print(new_label)
                self.data.extend(new_query)
                # print(len(self.data))
                self.label.extend(new_label)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        text = self.data[index]
        label = self.label[index]
        # print(text)
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,  # 根据需求设置最大长度
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze()
        },label


#
# # 定义数据文件路径和RoBERTa Tokenizer
# file_path = '../data/training_data/train_test.jsonl'
# tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
# model = RobertaForSequenceClassification.from_pretrained("roberta-base")
# # 创建数据集和数据加载器
# dataset = MyDataset(file_path, tokenizer)
# print(len(dataset))
# # print()
# dataloader = DataLoader(dataset, batch_size=5, shuffle=False)
# sample_num=0
# for encoded_input,label in dataloader:
#     sample_num+=len(encoded_input)
# print(sample_num)
#
#     # input_ids = encoded_input["input_ids"]
#     # # print(input_ids)
#     # attention_mask = encoded_input["attention_mask"]
#     #
#     # # Perform sequence classification
#     # outputs = model(input_ids, attention_mask=attention_mask)
#     #
#     # # Get the predicted class probabilities
#     # probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
#     #
#     #
#     # predicted_labels = torch.argmax(probs, dim=1)
#     # # print(probs)
#     #
#     # # Print the predicted probabilities and the predicted class
#     # # predicted_class = torch.argmax(probs, dim=-1).item()
#     # print("Predicted probabilities:", probs)
#     # print("Predicted class:", predicted_labels)
#
