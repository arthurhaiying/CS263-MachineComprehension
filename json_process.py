import jsonlines
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.utils.data import Dataset, DataLoader
from transformers import AdamW
from torch.nn import BCEWithLogitsLoss
from sklearn.metrics import accuracy_score
import torch.nn.functional as F


PLACE_HOLDER = '@placeholder'
def substitute(question, opt):
    question = question.replace(PLACE_HOLDER, opt)
    return question
def json_iter(path):
    with open(path, mode='r', encoding='utf-8') as f:
        reader = jsonlines.Reader(f)
        new_query=[]
        new_label=[]
        for instance in reader:
            article = instance["article"]
            question = instance["question"]
            opt0, opt1, opt2, opt3, opt4 = instance["option_0"], instance["option_1"], \
                instance["option_2"], instance["option_3"], instance["option_4"]
            opts = [opt0, opt1, opt2, opt3, opt4]
            label = instance["label"]
            instance_tuple = (article, question, opts, label)

            transform_binary(instance_tuple,new_label, new_query)
        return new_query, new_label
            # yield (article, question, opts, label)


def transform_binary(instance,new_label, new_query):
    article, question, opts, label = instance

    # scores = [0.0] * len(opts)
    # scores[label] = 1.0
    questions = [substitute(question, opt) for opt in opts]

    for i in range(len(questions)):
        Q_i = questions[i]
        seq_i =Q_i+'[SEP]'+article
        label_i = 1 if label==i else False
        new_label.append(label_i)
        new_query.append(seq_i)

def seq2tensor(sequence,
               max_seq_length=512, checkpoint = "bert-base-cased"):
    # sequences = ["I've been waiting for a HuggingFace course my whole life.", "So have I!"]

    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    # Returns PyTorch tensors
    tokens = tokenizer(sequence, truncation=True,padding='max_length', max_length=max_seq_length, return_tensors="pt")
    # input_ids=tokens['input_ids']
    # print("Input IDs:", input_ids)
    # return input_ids
    # print(type(tokens))

    return tokens

class MyBinaryDataset(Dataset):
    def __init__(self,X,Y):
        self.X = X
        self.Y = Y
        return

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        return self.X[index], self.Y[index]
def build_binary_dataset(json_path, max_seq_length=512, tokenizer="bert-base-cased"):
    query_seqs, binary_labels = json_iter(json_path)
    X_tokens = seq2tensor(query_seqs, max_seq_length=max_seq_length, checkpoint = tokenizer)
    # Y_tensor = F.one_hot(torch.tensor(binary_labels) ,num_classes = 2)
    Y_tensor = torch.tensor(binary_labels).long()
    # binary_dataset = MyBinaryDataset(X_tensor,Y_tensor)
    binary_dataset = MyBinaryDataset(X_tokens, Y_tensor)
    return binary_dataset


if __name__=='__main__':
    json_path = ("./data/training_data/train_test.jsonl")
    binary_dataset = build_binary_dataset(json_path)
    binary_dataloader = DataLoader(binary_dataset, batch_size=5, shuffle=True)
    checkpoint = "distilbert-base-uncased-finetuned-sst-2-english"
    model = AutoModelForSequenceClassification.from_pretrained(checkpoint)
    model.train()  # set model to training mode.
    for batch in binary_dataloader:
        x, y = batch
        print(x)
        out = model(x)  # shape (batch_size, n_classes)
        # print(out.logits.shape)