
# https://github.com/ai-forever/mgpt

from transformers import GPT2LMHeadModel, GPT2Tokenizer

# tokenizer = GPT2Tokenizer.from_pretrained("sberbank-ai/mGPT")
# model = GPT2LMHeadModel.from_pretrained("sberbank-ai/mGPT")

tokenizer = GPT2Tokenizer.from_pretrained("ai-forever/rugpt3small_based_on_gpt2")
model = GPT2LMHeadModel.from_pretrained("ai-forever/rugpt3small_based_on_gpt2")

text = "Владимир Владимирович родился в "
input_ids = tokenizer.encode(text, return_tensors="pt")
out = model.generate(
        input_ids, 
        min_length=20, 
        max_length=100, 
        eos_token_id=5, 
        pad_token=1,
        top_k=10,
        top_p=0.0,
        no_repeat_ngram_size=5
)
generated_text = list(map(tokenizer.decode, out))[0]
print(generated_text)