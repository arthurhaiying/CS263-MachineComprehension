import torch
from transformers import RobertaTokenizer,RobertaForMaskedLM
from torch.optim import AdamW
from torch.utils.data import DataLoader
import torch.nn as nn
from process import ClozeDataset
from sklearn.metrics import accuracy_score
import numpy as np


def evaluate(model, train_loader,device, k=5):
    model.eval()  # set model to training mode.

    total_loss = 0
    y_true = []
    y_pred = []
    sample_num = 0
    correct_topk = 0
    correct_num=0

    for inputs, opts, answers, mask_indices in train_loader:

        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]

        outputs = model(input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        batch_logits = logits[torch.arange(len(logits)), mask_indices]


        _, top_k_indices = torch.topk(batch_logits, k, dim=1)  # 获取预测结果中的 Top-k 索引
        _, top_1_indices = torch.topk(batch_logits, 1, dim=1)  # 获取预测结果中的 Top-k 索引
        # print(top_k_indices)
        correct = torch.sum(top_1_indices == answers.view(-1, 1), dim=1)
        correct_num += torch.sum(correct > 0).item()
        correct_k = torch.sum(top_k_indices == answers.view(-1, 1), dim=1)  # 比较 Top-k 索引与真实标签
        correct_topk += torch.sum(correct_k > 0).item()
        sample_num+=len(answers)


        # correct = torch.sum(top_k_indices == answers.view(-1, 1), dim=1)  # 比较 Top-k 索引与真实标签
        # correct_topk += torch.sum(correct > 0).item()

        # Calculate metrics for the epoch
    # accuracy = accuracy_score(y_true, y_pred)
    topk_acc = correct_topk / sample_num
    correctness = correct_num / len(train_loader)
    return topk_acc, correctness
def train(model, train_loader, optimizer, loss_fn, device,k=5):
    model.train()  # set model to training mode.

    total_loss = 0
    y_true = []
    y_pred = []
    sample_num=0
    correct_topk=0

    for inputs, opts, answers, mask_indices in train_loader:
        # inputs = inputs.to(device)
        # labels = labels.to(device)
        optimizer.zero_grad()
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]

        # Forward pass
        # outputs = model(inputs)
        outputs = model(input_ids, attention_mask=attention_mask)
        # logits = outputs.logits.squeeze()
        logits = outputs.logits
        # print(logits)
        # print(logits.shape)
        batch_logits = logits[torch.arange(len(logits)), mask_indices]
        # Compute the loss
        # print(batch_logits.shape)

        loss = loss_fn(batch_logits,answers)
        total_loss += loss.item()

        # Backward pass and optimization
        loss.backward()
        optimizer.step()
        # print(answers)

        _, top_k_indices = torch.topk(batch_logits, k, dim=1)  # 获取预测结果中的 Top-k 索引
        correct = torch.sum(top_k_indices == answers.view(-1, 1), dim=1)  # 比较 Top-k 索引与真实标签
        # print(correct)

        correct_topk += torch.sum(correct > 0).item()
        sample_num+=len(answers)

        if sample_num%500==0:
            print(str(sample_num)+"samples, Loss:" + str(loss))


        # Calculate metrics for the epoch
    # accuracy = accuracy_score(y_true, y_pred)
    # print('samplenum',sample_num)
    # print('len',len(train_loader))
    topk_acc = correct_topk/sample_num
    avg_loss = total_loss / len(train_loader)
    return topk_acc, avg_loss

def train_model(model, train_loader, val_loader, test_loader, optimizer, loss_fn, device, num_epochs=10):
    # best_val_accuracy = 0.0
    best_val_corr=0.0
    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs}")
        print("-" * 10)
        # Training
        train_accuracy, train_loss = train(model, train_loader, optimizer, loss_fn, device)
        print(f"Train Loss: {train_loss:.4f} | Train Accuracy: {train_accuracy:.4f}")

        # Validation
        val_accuracy, val_corr = evaluate(model, val_loader, device)
        print(f"Validation Accuracy: {val_accuracy:.4f} | Val Correctness:{val_corr:.4f} ")

        # Check if the current model has the best validation accuracy
        # if val_accuracy > best_val_accuracy:
        #     best_val_accuracy = val_accuracy
        if val_corr > best_val_corr:
            best_val_corr = val_corr
            torch.save(model.state_dict(), "best_model.pt")
            print("Best model saved!")

        print()

    print("Training complete.")

    # Load the best model and evaluate on the test set
    model.load_state_dict(torch.load("best_model.pt"))
    test_accuracy, test_corr = evaluate(model, test_loader, device)
    # print(f"Test Accuracy: {test_accuracy:.4f}")

    print(f"Test Accuracy: {test_accuracy:.4f} | Test Correctness:{test_corr:.4f}")


if __name__ == '__main__':
    checkpoint = "roberta-base"
    tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
    model = RobertaForMaskedLM.from_pretrained('roberta-base')
    max_len=512
    # ############### loading data
    train_json_path = ("../data/training_data/train_test.jsonl")
    val_json_path = ("../data/training_data/train_test.jsonl")
    test_json_path = ("../data/training_data/train_test.jsonl")
    train_batch_size=50

    # train_json_path = ("../data/training_data/Task_1_train.jsonl")
    # val_json_path = ("../data/training_data/Task_1_dev.jsonl")
    # test_json_path = ("../data/trail_data/Task_1_Imperceptibility.jsonl")
    #


    train_binary_dataset = ClozeDataset(train_json_path, tokenizer,max_len=max_len)
    train_loader = DataLoader(train_binary_dataset, batch_size=train_batch_size, shuffle=False)

    val_binary_dataset = ClozeDataset(val_json_path, tokenizer,max_len=max_len)
    val_loader = DataLoader(val_binary_dataset, batch_size=5, shuffle=False)

    test_binary_dataset = ClozeDataset(test_json_path, tokenizer,max_len=max_len)
    test_loader = DataLoader(test_binary_dataset, batch_size=5, shuffle=False)


    # ######################### hyper prm setting

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # batch_size = 5
    learning_rate = 1e-5
    num_epochs = 10

    optimizer = AdamW(model.parameters(), lr=learning_rate)
    # weights = torch.tensor([1.0, 4.0])
    # weights = torch.tensor(compute_class_weights(train_binary_dataset.label)).float()
    # print(weights)
    loss_fn = nn.CrossEntropyLoss()

    train_model(model, train_loader, val_loader, test_loader, optimizer, loss_fn, device, num_epochs=num_epochs)



