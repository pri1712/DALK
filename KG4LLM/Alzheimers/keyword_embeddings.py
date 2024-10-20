import os
import json
import pandas as pd
from tqdm import tqdm
import torch
!pip install sentence_transformers
from sentence_transformers import SentenceTransformer
import pickle

os.environ['hftoken'] = ''

df = pd.read_csv('KG4LLM/Alzheimers/train_s2s.txt', sep='\t', header=None, names=['head', 'relation', 'tail'])
entity = set()
for _, item in tqdm(df.iterrows()):
    entity.add(item[0])
    entity.add(item[2])

entity=list(entity)
print("Number of entities:",len(entity))

entity_words = set()
for file in os.listdir('KG4LLM/Alzheimers/result_ner'):
    dataset = json.load(open(os.path.join('KG4LLM/Alzheimers/result_ner', file)))
    for item in dataset:
        k_list = item['entity'].split('\n')
        for k in k_list:
            try:
                k = k.split('.')[1].strip() 
                entity_words.add(k)
            except:
                print(k)
                continue

keywords = list(entity_words)
print("Number of keywords:"len(keywords))

model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
#model.to("cuda")

embeddings = model.encode(entity, batch_size=1024, show_progress_bar=True, normalize_embeddings=True)
entity_emb_dict = {
    "entities": entity,
    "embeddings": embeddings,
}

with open("Alzheimers/entity_embeddings.pkl", "wb") as f:
    pickle.dump(entity_emb_dict, f)
print("Successfully encoded all entities")

embeddings = model.encode(keywords, batch_size=1024, show_progress_bar=True, normalize_embeddings=True)
keyword_emb_dict = {
    "keywords": keywords,
    "embeddings": embeddings,
}
with open("KG4LLM/Alzheimers/keyword_emb.pkl", "wb") as f:
    pickle.dump(keyword_emb_dict, f)
print("Successfully encoded all keywords")
