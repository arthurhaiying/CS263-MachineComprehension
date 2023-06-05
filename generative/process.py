import torch
from transformers import RobertaTokenizer, RobertaForSequenceClassification
from torch.utils.data import Dataset, DataLoader

import json
PLACE_HOLDER = '@placeholder'
def json2tuple(json_instance):

    article = json_instance["article"]
    question = json_instance["question"]
    opt0, opt1, opt2, opt3, opt4 = json_instance["option_0"], json_instance["option_1"], \
            json_instance["option_2"], json_instance["option_3"], json_instance["option_4"]
    opts = [opt0, opt1, opt2, opt3, opt4]
    # answer = opts[json_instance["label"]]
    label = json_instance["label"]
    # print(opts)
    instance_tuple = (article, question, opts, label)

    return instance_tuple

def transform(instance):
    article, question, opts, label = instance
    question = question.replace(PLACE_HOLDER, '<mask>')
    query = question + '</s></s>' +article
    return query, opts, label
class ClozeDataset(Dataset):
    def __init__(self, file_path, tokenizer,max_len=512):
        self.data = []
        self.mask_index = []
        self.opts = []
        self.label = []
        self.tokenizer = tokenizer
        self.max_len = max_len

        # 从JSONL文件中读取数据
        with open(file_path, 'r') as file:
            for line in file:
                json_obj = json.loads(line)
                instance_tuple = json2tuple(json_obj)
                query, opts, label = transform(instance_tuple)
                self.data.append(query)
                self.opts.append(opts)
                self.label.append(label)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        text = self.data[index]
        opts = self.opts[index]
        label = self.label[index]

        text_encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,  # 根据需求设置最大长度
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        # mask_token_index = torch.where(text_encoding == tokenizer.mask_token_id)[1]
        opts_token_id = [self.tokenizer.encode(opt, add_special_tokens=False)[0] for opt in opts]
        # answer_token_id = tokenizer.encode(answer, add_special_tokens=False)[0]
        answer_token_id = opts_token_id[label]
        inputs = self.tokenizer.encode_plus(text, add_special_tokens=True, padding='max_length',
                                            truncation=True, max_length=self.max_len)
        input_ids = inputs['input_ids']

        mask_index =input_ids.index(self.tokenizer.mask_token_id)
        # print(opts)
        # print('opts',opts_token_id)
        # # print(answer)
        # print('answer', answer_token_id)

        return {
            'input_ids': text_encoding['input_ids'].squeeze(),
            'attention_mask': text_encoding['attention_mask'].squeeze()
        }, opts_token_id, answer_token_id, mask_index


if __name__=='__main__':
    # 定义数据文件路径和RoBERTa Tokenizer
    file_path = '../data/training_data/train_test.jsonl'
    tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
    model = RobertaForSequenceClassification.from_pretrained("roberta-base")
    # 创建数据集和数据加载器
    dataset = ClozeDataset(file_path, tokenizer)
    print(len(dataset))
    # print()
    dataloader = DataLoader(dataset, batch_size=5, shuffle=False)
    print(len(dataloader))
    sample_num=0
    for x, opts, answer, mask_index in dataloader:
        sample_num+=len(x)
        # print(x, opts, answer,mask_index)