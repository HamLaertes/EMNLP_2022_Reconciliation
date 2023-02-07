from transformers import BertModel, BertTokenizer
import json
import numpy as np
import sklearn.manifold as manifold
from sklearn.decomposition import PCA
import torch
import torch.nn.functional as F
from tqdm import tqdm
import matplotlib.pyplot as plt
import random

with open("word_frequency.json", "r") as f:
    data = json.load(f)
print("Successful Load Data-File...")

# load the pre-trained word_vectors
bert_embedding = BertModel.from_pretrained("bert-base-uncased").embeddings
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
print("Successful Load Pre-trained Glove File...Start Mapping Words to Vectors...")
word_vectors = []
word_frequency = []
for d in tqdm(data):
    word = list(d.keys())[0].strip()
    word = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(word))
    word = torch.tensor(word).long()
    word_ebd = bert_embedding.word_embeddings(word).detach()
    word_ebd = word_ebd.mean(dim=0).unsqueeze(0)
    if torch.isnan(word_ebd).any():
        continue
    word_vectors.append(word_ebd)
    word_frequency.append(list(d.values())[0])
word_vectors = torch.cat(word_vectors, dim=0)
gold_vector = torch.ones(word_vectors.size()[-1], 1)

print("Successful Map Words to Vectors..Start Drawing Figures and Output...")
word_frequency = np.array(word_frequency)
word_vectors = F.normalize(word_vectors, dim=-1)
gold_vector = F.normalize(gold_vector, dim=0)
word_vectors = torch.matmul(word_vectors, gold_vector).squeeze().numpy()
order = np.argsort(word_frequency)
word_frequency = word_frequency[order]
word_vectors = word_vectors[order]
fig = plt.figure(figsize=(10, 10))
plt.scatter(x=word_vectors[:55000], y=word_frequency[:55000], alpha=1,
            c=word_frequency[:55000], cmap='rainbow')
plt.colorbar()
plt.savefig("bert_angle_frequency.jpg")
plt.close()

fig = plt.figure(figsize=(10, 10))
plt.scatter(x=word_vectors[:55000], y=[0 for _ in range(55000)], alpha=1,
            c=word_frequency[:55000], cmap='rainbow')
plt.colorbar()
plt.savefig("bert_angle_frequency_flatten.jpg")
plt.close()