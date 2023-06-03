from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
import torch
import jsonlines
import torch.nn as nn


from SiameseLSTM import Siamese_lstm

# experiment settings
train_data_path = "data/training_data/Task_1_dev.jsonl"
PLACE_HOLDER = '@placeholder'
PAD = "<PAD>"

embed_size = 256
hidden_size = 128
num_layers = 2
batch_size = 5
epoch_num = 5

# dataloader
tokenizer = get_tokenizer('basic_english')


def train_iter(path):
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
def yield_tokens(train_iter, tokenizer):
    for article, query, opts, _ in train_iter:
        opts = " ".join(opts)
        text = article + " " + query + " " + opts
        tokens = tokenizer(text)
        yield tokens


yield_tokens_iter = yield_tokens(train_iter(train_data_path), tokenizer)
vocab = build_vocab_from_iterator(yield_tokens_iter, specials=[PAD])
vocab.set_default_index(vocab[PLACE_HOLDER])
vocab_size = len(vocab)
PAD_ID = vocab[PAD]
PLACE_HOLDER_ID = vocab[PLACE_HOLDER]


def sent2ids(sent):
    tokens = tokenizer(sent)
    return vocab(tokens)


def substitute(query, opt):
    query = query.replace(PLACE_HOLDER, opt)
    return query


# transform each instance (article, query, 5 opts, label) into five instances (X1 , X2, score)
# TODO: padding
def transform(instance):
    article, query, opts, label = instance
    scores = [0.0] * len(opts)
    scores[label] = 1.0
    queries = [substitute(query, opt) for opt in opts]
    article = sent2ids(article)
    queries = [sent2ids(query) for query in queries]
    articles = [article] * 5
    X1 = torch.tensor(articles, dtype=torch.int64)
    X2 = torch.tensor(queries, dtype=torch.int64)
    # Y = torch.tensor(scores, dtype=torch.float64)
    Y = torch.tensor(scores, dtype=torch.float32)

    return X1, X2, Y


# For now, batch size = 5
def train_dataloader(train_iter):
    for instance in train_iter:
        yield transform(instance)


def main():
    train_loader = train_iter(train_data_path)
    model = Siamese_lstm(embed_size, hidden_size, num_layers, batch_size, vocab_size)
    optimizer= torch.optim.Adam(model.parameters())
    for epoch in range(epoch_num):
        # training
        model.train()
        cnt=0
        for instance in train_loader:
            cnt+=1
            if cnt%20==0:
                print(cnt)
            optimizer.zero_grad()
            batch = transform(instance)
            X1, X2, Y = batch
            score = model(X1, X2)
            loss = nn.MSELoss()
            l=loss(score, Y)
            # print(l)
            l.backward()
            optimizer.step()

        # testing
        model.eval()
        test_loader = train_iter(train_data_path)
        ns=0
        nc=0
        with torch.no_grad():
            print('eval')
            for instance in test_loader:
                batch = transform(instance)
                X1, X2, Y = batch
                score = model(X1, X2)
                ns+=1
                nc+=(score.argmax() == Y.argmax())
                #print(f"instance {ns}, predict {score.argmax()}, label {Y.argmax()}")

            print(f"nc: {nc}, ns: {ns}")
            acc = (nc/ns).detach().cpu().numpy()
            print('======epoch:', epoch,' , acc:',acc)
    

if __name__ == '__main__':
    main()
    # instance0 = next(train_iter(train_data_path))
    # article0, query0, opts0, label0 = instance0
    # Q0 = tokenizer(query0)
    # print(f"query: {Q0}\n")
    # vec0 = sent2ids(query0)
    # print(f"query tokens: {vec0}\n")
    # print(f"vocab size: {vocab_size}")
    # print(f"placeholder id: {PLACE_HOLDER_ID}")
    # batch0 = transform(instance0)
    # X1, X2, Y = batch0
    # # print("X1: ", X1)
    # print("X1: ", X1.shape)
    # # print(X1.type)
    # # print("X2: ", X2)
    # print("X2: ", X2.shape)
    # # print("score: ", Y)
    #
    # print("score: ", Y.shape)
    # model = Siamese_lstm(embed_size, hidden_size, num_layers, batch_size, vocab_size)
    # score = model(X1, X2)
    # print(f"score: {score.size()}", score)


# training metric: mean square error
# testing metric: argmax, label groundtruth
