
import os
from transformers import (AutoTokenizer, 
                          AutoModelForCausalLM)

finetuned_path = os.path.join("model_out")
model_name_or_path = finetuned_path # Указываем путь до папки с нашей моделью
tokenizer = AutoTokenizer.from_pretrained(model_name_or_path) # Активируем токенизатор
gpt = AutoModelForCausalLM.from_pretrained(model_name_or_path) # Активируем модель

prompt = 'упрощенная декларация' # Начальные слова, которые подаются модели. С них и будет начинаться генерируемый текст
input_ids = tokenizer.encode(prompt, return_tensors="pt") # С помощью токенизатора преобразуем начальные слова в формат для модели

generated_text_samples = gpt.generate(
    input_ids, 
    max_length=100,
    num_return_sequences=1,
    num_beams=2,
    no_repeat_ngram_size=3,
    repetition_penalty=1.5,
    top_p=1.,
    temperature=0.8,
    do_sample=True,
    top_k=125,
    early_stopping=True
)

# Выводим результат генерации
for i, beam in enumerate(generated_text_samples):
  print("{}: {}".format(i,tokenizer.decode(beam, skip_special_tokens=False).split('<|endoftext|>')[0]))