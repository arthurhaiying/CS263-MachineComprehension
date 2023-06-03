import torch
from pytorch_pretrained_bert import BertTokenizer, BertModel, BertForMaskedLM

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
text = 'From Monday to Friday most people are busy working or studying, '\
       'but in the evenings and weekends they are free and _ themselves.'
tokenized_text = tokenizer.tokenize(text)

masked_index = tokenized_text.index('_')
tokenized_text[masked_index] = '[MASK]'

candidates = ['love', 'work', 'enjoy', 'play']
candidates_ids = tokenizer.convert_tokens_to_ids(candidates)

indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)

segments_ids = [0] * len(tokenized_text)

tokens_tensor = torch.tensor([indexed_tokens])
segments_tensors = torch.tensor([segments_ids])

language_model = BertForMaskedLM.from_pretrained('bert-base-uncased')
language_model.eval()

predictions = language_model(tokens_tensor, segments_tensors)
predictions_candidates = predictions[0, masked_index, candidates_ids]
answer_idx = torch.argmax(predictions_candidates).item()

print(f'The most likely word is "{candidates[answer_idx]}".')