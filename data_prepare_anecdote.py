import os
import pandas as pd

anek_path = os.path.join("data", "anek.txt")
l = []

with open(anek_path, 'r', encoding="utf-8") as anek_text:
    for line in anek_text:
        if line.strip():
            l.append(line.strip())
# Создадим датафрейм на основе получившегося списка анекдотов

df = pd.DataFrame(l)
df.rename(columns={ df.columns[0]: "clean_text"}, inplace = True)
print(df)

texts = '<|endoftext|>'.join([text.replace('\n\n\n\n','') for text in df['clean_text']])

texts_path = os.path.join("data", 'texts_dataset.txt')
with open(texts_path, "w", encoding='utf-8') as d:
    d.write(texts)
    
# Посмотрим, что они корректно сохранились
with open(texts_path, "r", encoding='utf-8') as d:
  opened_texts = d.read()

print(opened_texts[:1000])