import os
import datasets
from transformers import GPT2Tokenizer

tokenizer = GPT2Tokenizer.from_pretrained("ai-forever/rugpt3small_based_on_gpt2")
data = datasets.load_from_disk(os.path.join("data", "english_quotes.dataset"))
print(data['train']['quote'])

data = data.map(lambda samples: tokenizer(samples['quote']), batched=True)
# print(data['train']['input_ids'])