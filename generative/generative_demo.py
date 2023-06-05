import torch
from transformers import RobertaTokenizer, RobertaForMaskedLM

# 初始化RoBERTa模型和tokenizer
tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
model = RobertaForMaskedLM.from_pretrained('roberta-base')

# 文本示例
text = "This is a sample sentence to calculate the probability of a specific word."

# 指定的单词
target_word = "specific"

# 准备输入
encoded_input = tokenizer.encode_plus(text, add_special_tokens=True, return_tensors='pt')

# 获取input_ids和attention_mask
input_ids = encoded_input['input_ids']
attention_mask = encoded_input['attention_mask']

# 调用模型
with torch.no_grad():
    outputs = model(input_ids=input_ids, attention_mask=attention_mask)

# 获取预测的logits
logits = outputs.logits

# 获取目标单词的索引
target_word_index = tokenizer.encode(target_word, add_special_tokens=False)[0]

# 计算概率
softmax = torch.nn.Softmax(dim=-1)
probabilities = softmax(logits[0, :, :])

# 输出概率
print(f"Probability of '{target_word}': {probabilities[0, target_word_index].item()}")
