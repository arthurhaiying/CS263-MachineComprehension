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
hidden_size = 64
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
            question = instance["question"]
            opt0, opt1, opt2, opt3, opt4 = instance["option_0"], instance["option_1"], \
                instance["option_2"], instance["option_3"], instance["option_4"]
            opts = [opt0, opt1, opt2, opt3, opt4]
            label = instance["label"]
            yield (article, question, opts, label)


# build vocabulary
def yield_tokens(train_iter, tokenizer):
    for article, question, opts, _ in train_iter:
        opts = " ".join(opts)
        text = article + " " + question + " " + opts
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


def substitute(question, opt):
    question = question.replace(PLACE_HOLDER, opt)
    return question


# transform each instance (article, question, 5 opts, label) into five instances (X1 , X2, score)
# TODO: padding
def transform(instance):
    article, question, opts, label = instance
    scores = [0.0] * len(opts)
    scores[label] = 1.0
    questions = [substitute(question, opt) for opt in opts]
    article = sent2ids(article)
    questions = [sent2ids(question) for question in questions]
    articles = [article] * 5
    X1 = torch.tensor(articles, dtype=torch.int64)
    X2 = torch.tensor(questions, dtype=torch.int64)
    # Y = torch.tensor(scores, dtype=torch.float64)
    Y = torch.tensor(scores, dtype=torch.float32)

    return X1, X2, Y


# For now, batch size = 5
def train_dataloader(train_iter):
    for instance in train_iter:
        yield transform(instance)


def train():
    train_loader = train_iter(train_data_path)
    model = Siamese_lstm(embed_size, hidden_size, num_layers, batch_size, vocab_size)
    optimizer= torch.optim.Adam(model.parameters())
    for epoch in range(epoch_num):
        model.train()
        cnt=0
        for instance0 in train_loader:
            cnt+=1
            if cnt%20==0:
                print(cnt)
            optimizer.zero_grad()
            article0, question0, opts0, label0 = instance0
            Q0 = tokenizer(question0)
            # print(f"Question: {Q0}\n")
            vec0 = sent2ids(question0)
            # print(f"Question tokens: {vec0}\n")
            # print(f"vocab size: {vocab_size}")
            # print(f"placeholder id: {PLACE_HOLDER_ID}")
            batch0 = transform(instance0)
            X1, X2, Y = batch0
            score = model(X1, X2)
            loss = nn.MSELoss()
            l=loss(score, Y)
            # print(score.dtype)
            # print(Y.dtype)
            l.backward()
            optimizer.step()
        model.eval()
        ns=0
        nc=0
        with torch.no_grad():
            for instance0 in train_loader:
                batch0 = transform(instance0)
                X1, X2, Y = batch0
                score = model(X1, X2)
                ns+=len(Y)
                nc+=(score.max(1)[1]==Y).detach().cpu().numpy().sum()
                print('epoch:',epoch,' ,acc:',nc/ns)
                # loss = nn.MSELoss(score, Y)
                # loss.backward()
                # optimizer.step()

if __name__ == '__main__':
    train()
    # instance0 = next(train_iter(train_data_path))
    # article0, question0, opts0, label0 = instance0
    # Q0 = tokenizer(question0)
    # print(f"Question: {Q0}\n")
    # vec0 = sent2ids(question0)
    # print(f"Question tokens: {vec0}\n")
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
