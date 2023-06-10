from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import torch
import jsonlines
import torch.nn as nn


# from ClozeLSTM import ClozeLSTM

# experiment settings
train_data_path = "./newdata/train/Task_2_train.jsonl"
val_data_path = "./newdata/val/Task_2_val.jsonl"
test_data_path = "./newdata/test/Task_2_test.jsonl"
UNK = "<UNK>" # unseen word in training set
PLACE_HOLDER = '@placeholder'
PAD = "<PAD>"
SEP = "<SEP>"


experiment_config = {
    'training':{
        'num_epochs': 10,
        'batch_size': 10,
        'maxdataset_size': 1000,
        'max_seq_len':512
    },
        
    'model':{
        'embed_size': 256,
        'hidden_size': 64,
        'num_layers': 2, 
        'dropout': 0.3 
    },   
}


# create dataset
tokenizer = get_tokenizer('basic_english')

def data_iter(path):
    with open(path, mode='r', encoding='utf-8') as f:
        reader = jsonlines.Reader(f)
        for instance in reader:
            article = instance["article"]
            query = instance["question"]
            opt0, opt1, opt2, opt3, opt4 = instance["option_0"], instance["option_1"], \
                instance["option_2"], instance["option_3"], instance["option_4"]
            opts = [opt0, opt1, opt2, opt3, opt4]
            label = instance["label"]
            yield (article, query, opts, label)


# build vocabulary
def yield_tokens(data_iter, tokenizer):
    for article, query, opts, _ in data_iter:
        opts = " ".join(opts)
        text = article + " " + query + " " + opts
        tokens = tokenizer(text)
        yield tokens

yield_tokens_iter = yield_tokens(data_iter(train_data_path), tokenizer)
vocab = build_vocab_from_iterator(yield_tokens_iter, specials=[UNK,PAD,PLACE_HOLDER,SEP])
PAD_ID = vocab[PAD]
UNK_ID = vocab[UNK]
PLACE_HOLDER_ID = vocab[PLACE_HOLDER] # id 1
SEP_ID = vocab[SEP] # id 2
vocab.set_default_index(UNK_ID)
vocab_size = len(vocab)
experiment_config["model"]["vocab_size"] = vocab_size
experiment_config["model"]["pad"] = PAD_ID


def word2id(word):
    return vocab[word]

def words2ids(words):
    return vocab(words)

def sent2ids(sent):
    tokens = tokenizer(sent)
    return vocab(tokens)

def substitute(query, opt):
    query = query.replace(PLACE_HOLDER, opt)
    return query

# TODO: padding
# assume batch size = 1
def transform(instance):
    article, question, opts, label = instance
    qa = sent2ids(question)+[SEP_ID]+sent2ids(article)
    max_seq_len = experiment_config["training"]["max_seq_len"]
    #if len(qa)>max_seq_len:
        #qa = qa[:max_seq_len]
    X = torch.tensor(qa,dtype=torch.int64)
    # assume each opt is single word
    opt = opts[label]
    Y = torch.tensor(word2id(opt), dtype=torch.int64) # id of positive word
    opts = torch.tensor(words2ids(opts), dtype=torch.int64) # id of opts
    label = torch.tensor(label, dtype=torch.int64)
    # X = X.view(1, -1) # (1, L)
    # Y = Y.view(1) # (1,)
    # opts = opts.view(1, -1) # (1, 5)
    # label = label.view(1, -1) # (1, 1)
    return X, Y, opts, label


# TODO: padding
# assume batch size = 1
def transform2(instance):
    article, question, opts, label = instance
    qa = sent2ids(question)+[SEP_ID]+sent2ids(article)
    X = torch.tensor(qa,dtype=torch.int64)
    # assume each opt is single word
    optP = opts[label]
    optsN = [o for o in opts if o != optP]
    YP = torch.tensor(word2id(optP), dtype=torch.int64)
    YN = torch.tensor(words2ids(optsN), dtype=torch.int64) 
    X = X.view(1, -1)
    YP = YP.view(1)
    YN = YN.view(1, -1)
    return X, YP, YN



class MyDataset(Dataset):
    """Re dataset."""

    def __init__(self, data_path, transform=None):
        self.data_path = data_path
        self.transform = transform

        self.data = []
        with open(data_path, mode='r', encoding='utf-8') as f:
            reader = jsonlines.Reader(f)
            for i, instance in enumerate(reader):
                max_dataset_size = experiment_config["training"]["maxdataset_size"]
                #if i>= max_dataset_size:
                    #break
                article = instance["article"]
                query = instance["question"]
                opt0, opt1, opt2, opt3, opt4 = instance["option_0"], instance["option_1"], \
                    instance["option_2"], instance["option_3"], instance["option_4"]
                opts = [opt0, opt1, opt2, opt3, opt4]
                label = instance["label"]
                instance = (article, query, opts, label)
                self.data.append(instance)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        instance = self.data[idx]
        if self.transform:
            X, Y, opts, label = self.transform(instance)
            return X, Y, opts, label
        else:
            return instance
        
# convert a list of samples into batch and padding
def collate(batch):
    X_list, Y_list, opts_list, label_list = [], [], [], []
    for X,Y,opts,label in batch:
        X_list.append(X)
        Y_list.append(Y)
        opts_list.append(opts)
        label_list.append(label)
    
    lengths = [len(X) for X in X_list]
    X_pad = pad_sequence(X_list, batch_first=True, padding_value=PAD_ID) # pad input to maxlen
    Y_ = torch.stack(Y_list)
    opts_ = torch.stack(opts_list)
    labels_ = torch.stack(label_list)
    return X_pad, Y_, opts_, labels_, lengths




# For now, batch size = 5
def evaluate(model, test_loader):
    model.eval()
    ns = 0
    nc = 0
    with torch.no_grad():
        print('eval')
        for batch in test_loader:
            X, Y, opts, labels, lengths = batch
            batch_size = len(X)
            output = model(X, lengths) # (N, V)
            output = torch.take_along_axis(output, opts, dim=1)
            Y_pred = torch.argmax(output, dim=1) # (N,)
            labels = labels.squeeze()
            ns += batch_size
            nc += torch.sum(Y_pred==labels)
            #print(f"instance {ns}, predict {score.argmax()}, label {Y.argmax()}")

        print(f"nc: {nc}, ns: {ns}", flush=True)
        acc = (nc/ns).detach().cpu().numpy()

    return acc






def main():
    train_dataset = MyDataset(train_data_path, transform=transform)
    test_dataset = MyDataset(test_data_path, transform=transform)
    val_dataset = MyDataset(val_data_path, transform=transform)
    model = ClozeLSTM(config=experiment_config)
    optimizer = torch.optim.Adam(model.parameters())
    num_epochs = experiment_config["training"]["num_epochs"]
    for epoch in range(num_epochs):
        print(f"start epoch: {epoch}")
        # training
        train_loader = DataLoader(train_dataset, 
                                  batch_size=experiment_config["training"]["batch_size"], 
                                  collate_fn=collate,
                                  shuffle=True)
        model.train()
        cnt=0
        for instance in train_loader:
            cnt+=1
            if cnt%20==0:
                print(cnt)
            X, Y, opts, label, lengths = instance
            optimizer.zero_grad()
            output = model(X, lengths)
            loss = nn.CrossEntropyLoss()
            l=loss(output, Y)
            # print(l)
            l.backward()
            optimizer.step()

        # testing
        train_loader = DataLoader(train_dataset, 
                                  batch_size=experiment_config["training"]["batch_size"], 
                                  collate_fn=collate,
                                  shuffle=True)
        test_loader = DataLoader(test_dataset, 
                                 batch_size=experiment_config["training"]["batch_size"], 
                                 collate_fn=collate,
                                 shuffle=True)
        val_loader = DataLoader(val_dataset, 
                                 batch_size=experiment_config["training"]["batch_size"], 
                                 collate_fn=collate,
                                 shuffle=True)
        train_acc = evaluate(model, train_loader)
        val_acc = evaluate(model, val_loader)
        test_acc = evaluate(model, test_loader)
        print(f"=== epoch: {epoch}, train acc: {train_acc}, val acc: {val_acc},test acc: {test_acc}\n")
    

if __name__ == '__main__':
    # instance0 = next(data_iter(train_data_path))
    # article0, query0, opts0, label0 = instance0
    # Q0 = tokenizer(query0)
    # print(f"query: {Q0}\n")
    # batch0 = transform2(instance0)
    # X, YP, YN = batch0
    # X = X.view(1, -1)
    # YP = YP.view(1, -1)
    # YN = YN.view(1, -1)
    # print(f"X: {X.size()}", X)
    # print(f"\nYP: {YP.size()}", YP)
    # print(f"\nYN: {YN.size()}", YN)
    # experiment_config["model"]["vocab_size"] = vocab_size
    # model = ClozeLSTM(config=experiment_config) 
    # output = model(X)
    # print(f"output: {output.size()}", output)
    # train_dataset = MyDataset(train_data_path, transform=transform)
    # train_dataloader = DataLoader(train_dataset, 
    #                               batch_size=experiment_config["training"]["batch_size"],
    #                               collate_fn=collate, 
    #                               shuffle=True)
    # X, Y, opts, label, lengths = next(iter(train_dataloader))
    # print(f"X: {X.size()}", X)
    # print(f"Y: {Y.size()}", Y)
    # print(f"opts {opts.size()}: ", opts)
    # print(f"label: {label.size()}", label )
    # print("lengths: ", lengths)
    print(vocab_size)
    main()
    