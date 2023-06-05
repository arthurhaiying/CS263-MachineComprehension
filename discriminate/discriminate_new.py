import torch
from transformers import RobertaTokenizer, RobertaForSequenceClassification
from torch.optim import AdamW
from torch.utils.data import DataLoader
import torch.nn as nn
from discriminate import batch_process
from sklearn.metrics import accuracy_score
import numpy as np
def train(model, train_loader, optimizer, loss_fn, device):
    model.train()  # set model to training mode.

    total_loss = 0
    y_true = []
    y_pred = []

    for inputs, labels in train_loader:
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
        # Compute the loss

        loss = loss_fn(logits, labels)
        total_loss += loss.item()

        # Backward pass and optimization
        loss.backward()
        optimizer.step()

        # Collect predictions and true labels for evaluation
        # predicted_labels = torch.sigmoid(logits) > 0.5
        predicted_labels = torch.argmax(logits, dim=1)
        y_true.extend(labels.tolist())
        y_pred.extend(predicted_labels.tolist())


        # Calculate metrics for the epoch
    accuracy = accuracy_score(y_true, y_pred)
    avg_loss = total_loss / len(train_loader)
    return accuracy, avg_loss

def evaluate(model, data_loader, device):
    model.eval()
    y_true = []
    y_pred = []
    nc=0
    ns=0

    with torch.no_grad():
        for inputs, labels in data_loader:
            # print(inputs)
            input_ids = inputs["input_ids"]
            attention_mask = inputs["attention_mask"]

            # Forward pass
            # outputs = model(inputs)
            outputs = model(input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            # print(logits)
            predicted_labels = torch.argmax(logits, dim=1)
            y_true.extend(labels.tolist())
            y_pred.extend(predicted_labels.tolist())

            ns += len(labels) / 5
            nc += (labels.argmax() == predicted_labels.argmax())


            # # Collect predictions and true labels for evaluation
            # predicted_labels = torch.sigmoid(logits) > 0.5
            # # print(predicted_labels)
            # y_true.extend(labels.tolist())
            # y_pred.extend(predicted_labels.tolist())

            # Calculate metrics
        accuracy = accuracy_score(y_true, y_pred)
        correctness = nc/ns

        return accuracy, correctness

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
        print(f"Validation Accuracy: {val_accuracy:.4f} | Val Correctness:{val_corr:.4f}")

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

def compute_class_weights(labels):
    class_counts = np.bincount(labels)
    total_samples = len(labels)
    class_weights = total_samples / (len(class_counts) * class_counts)
    return class_weights

if __name__ == '__main__':
    checkpoint = "roberta-base"
    tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
    model = RobertaForSequenceClassification.from_pretrained("roberta-base")
    max_len=512
    # ############### loading data
    train_json_path = ("../data/training_data/train_test.jsonl")
    val_json_path = ("../data/training_data/train_test.jsonl")
    test_json_path = ("../data/training_data/train_test.jsonl")

    # train_json_path = ("../data/training_data/Task_1_train.jsonl")
    # val_json_path = ("../data/training_data/Task_1_dev.jsonl")
    # test_json_path = ("../data/trail_data/Task_1_Imperceptibility.jsonl")
    #


    train_binary_dataset = batch_process.MyDataset(train_json_path, tokenizer,max_len=max_len)
    train_loader = DataLoader(train_binary_dataset, batch_size=5, shuffle=False)

    val_binary_dataset = batch_process.MyDataset(val_json_path, tokenizer,max_len=max_len)
    val_loader = DataLoader(val_binary_dataset, batch_size=5, shuffle=False)

    test_binary_dataset = batch_process.MyDataset(test_json_path, tokenizer,max_len=max_len)
    test_loader = DataLoader(test_binary_dataset, batch_size=5, shuffle=False)


    # ######################### hyper prm setting

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = 5
    learning_rate = 1e-5
    num_epochs = 10

    optimizer = AdamW(model.parameters(), lr=learning_rate)
    # weights = torch.tensor([1.0, 4.0])
    weights = torch.tensor(compute_class_weights(train_binary_dataset.label)).float()
    # print(weights)
    loss_fn = nn.CrossEntropyLoss(weight = weights)

    train_model(model, train_loader, val_loader, test_loader, optimizer, loss_fn, device, num_epochs=2)



