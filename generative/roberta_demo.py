from transformers import AutoModelForMaskedLM, AutoTokenizer
import torch

tokenizer = AutoTokenizer.from_pretrained('roberta-base')
model = AutoModelForMaskedLM.from_pretrained('roberta-base')

sequence = f"The world will end in {tokenizer.mask_token}" # "The world will end in <mask>"

input_seq = tokenizer.encode(sequence, return_tensors='pt') # tensor([[0, 133, 232, 40, 253, 11, 50264, 2]])
mask_token_index = torch.where(input_seq == tokenizer.mask_token_id)[1] # (tensor([0]), tensor([6])) - we only want the the 2nd dimension

token_logits = model(input_seq).logits
masked_token_logits = token_logits[0, mask_token_index, :]

top_5_tokens = torch.topk(masked_token_logits, 5, dim=1).indices[0].tolist()

print('sequence:', sequence)
print('input_seq:', input_seq)
print('mask_token_index:', mask_token_index)
print('token_logits:', token_logits)
print('masked_token_logits:', masked_token_logits)
print('top_5_tokens:', top_5_tokens)