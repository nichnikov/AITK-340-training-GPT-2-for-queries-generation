import os
import pandas as pd

query_df = pd.read_csv(os.path.join("data", "not_similar_queries.csv"), sep="\t")
print(query_df)
# queries = list(query_df["query"])

queries_for_training = '<|endoftext|>'.join([text.replace('\n\n\n\n','') for text in query_df["query"]])

texts_path = os.path.join("data", 'queries_dataset.txt')
with open(texts_path, "w", encoding='utf-8') as d:
    d.write(queries_for_training)
    
# Посмотрим, что они корректно сохранились
with open(texts_path, "r", encoding='utf-8') as d:
  opened_texts = d.read()

print(opened_texts[:1000])
