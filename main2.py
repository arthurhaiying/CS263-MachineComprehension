from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from torch.utils.data import Dataset, DataLoader
import torch
import jsonlines
import torch.nn as nn


from ClozeLSTM import ClozeLSTM

# experiment settings
train_data_path = "data/training_data/Task_1_train.jsonl"
test_data_path = "data/training_data/Task_1_dev.jsonl"
UNK = "<UNK>" # unseen word in training set
PLACE_HOLDER = '@placeholder'
#PAD = "<PAD>"
SEP = "<SEP>"


experiment_config = {
    'training':{
        'num_epochs': 10,
        'batch_size': 1
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
vocab = build_vocab_from_iterator(yield_tokens_iter, specials=[UNK,PLACE_HOLDER,SEP])
#PAD_ID = vocab[PAD]
UNK_ID = vocab[UNK]
PLACE_HOLDER_ID = vocab[PLACE_HOLDER] # id 1
SEP_ID = vocab[SEP] # id 2
vocab.set_default_index(UNK_ID)
vocab_size = len(vocab)
experiment_config["model"]["vocab_size"] = vocab_size


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
            for instance in reader:
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


# For now, batch size = 5
def evaluate(model, test_loader):
    model.eval()
    ns = 0
    nc = 0
    with torch.no_grad():
        print('eval')
        for instance in test_loader:
            batch = transform(instance)
            X, Y, opts, label = batch
            #batch_size = 1
            output = model(X)
            output = output[0] # (V,)
            opts = opts[0] # (5,)
            output = output[opts]
            Y_pred = output.argmax()
            ns += 1
            nc += (Y_pred==label[0])
            #print(f"instance {ns}, predict {score.argmax()}, label {Y.argmax()}")

        print(f"nc: {nc}, ns: {ns}", flush=True)
        acc = (nc/ns).detach().cpu().numpy()

    return acc



def main():
    train_dataset = MyDataset(train_data_path, transform=transform)
    train_loader = DataLoader(train_dataset, 
                              batch_size=experiment_config["training"]["batch_size"], 
                              shuffle=True)
    model = ClozeLSTM(config=experiment_config)
    optimizer = torch.optim.Adam(model.parameters())
    num_epochs = experiment_config["training"]["num_epochs"]
    for epoch in range(num_epochs):
        print(f"start epoch: {epoch}")
        # training
        model.train()
        cnt=0
        for instance in train_loader:
            cnt+=1
            if cnt%20==0:
                print(cnt)
            X, Y, opts, label = instance
            optimizer.zero_grad()
            output = model(X)
            loss = nn.CrossEntropyLoss()
            l=loss(output, Y)
            # print(l)
            l.backward()
            optimizer.step()

        # testing
        train_loader = data_iter(train_data_path)
        test_loader = data_iter(test_data_path)
        train_acc = evaluate(model, train_loader)
        test_acc = evaluate(model, test_loader)
        print(f"=== epoch: {epoch}, train acc: {train_acc}, test acc: {test_acc}\n")
    

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
    #                               shuffle=True)
    # X, Y, opts, label = next(iter(train_dataloader))
    # print(f"X: {X.size()}", X)
    # print(f"Y: {Y.size()}", Y)
    # print(f"opts: ", opts)
    # print(f"label: ", label )
    main()
    