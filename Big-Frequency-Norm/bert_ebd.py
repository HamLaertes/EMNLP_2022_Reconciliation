from transformers import BertModel, BertTokenizer
from transformers import RobertaModel, RobertaTokenizer
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
word_vectors = torch.cat(word_vectors, dim=0).numpy()

print("Successful Map Words to Vectors...Start PCA and TSNE...")
word_frequency = np.array(word_frequency)
order = np.argsort(word_frequency)
random.shuffle(order)
word_frequency = word_frequency[order]
word_vectors = word_vectors[order]
pca = PCA(n_components=2)
word_mean = np.average(word_vectors, axis=0)
pca.fit(word_vectors - word_mean)
word_vectors = pca.transform(word_vectors)

print("Successful Complete PCA...Start Drawing Figures and Output...")
order = np.argsort(word_frequency)
word_frequency = word_frequency[order]
word_vectors = word_vectors[order]
np.save("bert_raw.npy", np.array(word_vectors))
np.save("bert_c.npy", np.array(word_frequency))

# load the pre-trained word_vectors
roberta_embedding = RobertaModel.from_pretrained("roberta-base").embeddings
tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
print("Successful Load Pre-trained Glove File...Start Mapping Words to Vectors...")
word_vectors = []
word_frequency = []
for d in tqdm(data):
    word = list(d.keys())[0].strip()
    word = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(word))
    word = torch.tensor(word).long()
    word_ebd = roberta_embedding.word_embeddings(word).detach()
    word_ebd = word_ebd.mean(dim=0).unsqueeze(0)
    if torch.isnan(word_ebd).any():
        continue
    word_vectors.append(word_ebd)
    word_frequency.append(list(d.values())[0])
word_vectors = torch.cat(word_vectors, dim=0).numpy()

print("Successful Map Words to Vectors...Start PCA and TSNE...")
word_frequency = np.array(word_frequency)
order = np.argsort(word_frequency)
random.shuffle(order)
word_frequency = word_frequency[order]
word_vectors = word_vectors[order]
pca = PCA(n_components=2)
word_mean = np.average(word_vectors, axis=0)
pca.fit(word_vectors - word_mean)
word_vectors = pca.transform(word_vectors)

print("Successful Complete PCA...Start Drawing Figures and Output...")
order = np.argsort(word_frequency)
word_frequency = word_frequency[order]
word_vectors = word_vectors[order]
np.save("roberta_raw.npy", np.array(word_vectors))
np.save("roberta_c.npy", np.array(word_frequency))

#fig = plt.figure(figsize=(10, 10))
# plt.scatter(x=word_vectors[:55000,0], y=word_vectors[:55000,1], alpha=1,
#             c=word_frequency[:55000], cmap='rainbow')
# plt.colorbar()
# plt.savefig("bert_raw.jpg")
# plt.close()

# print("Normalize the Word Vectors by L2 Norm and Repeat the above Steps...")
# word_vectors = []
# word_frequency = []
# for d in tqdm(data):
#     word = list(d.keys())[0].strip()
#     word = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(word))
#     word = torch.tensor(word).long()
#     word_ebd = bert_embedding.word_embeddings(word).detach()
#     word_ebd = word_ebd.mean(dim=0).unsqueeze(0)
#     if torch.isnan(word_ebd).any():
#         continue
#     word_vectors.append(word_ebd)
#     word_frequency.append(list(d.values())[0])
# word_vectors = torch.cat(word_vectors, dim=0)
# word_vectors = F.normalize(word_vectors, dim=-1).numpy()
#
# word_frequency = np.array(word_frequency)
# order = np.argsort(word_frequency)
# random.shuffle(order)
# word_frequency = word_frequency[order]
# word_vectors = word_vectors[order]
# print("Successful Map Words to Vectors...Start PCA...")
# pca = PCA(n_components=2)
# word_mean = np.average(word_vectors, axis=0)
# pca.fit(word_vectors - word_mean)
# word_vectors = pca.transform(word_vectors)
#
# order = np.argsort(word_frequency)
# word_frequency = word_frequency[order]
# word_vectors = word_vectors[order]
# print("Successful Complete PCA...Start Drawing Figures and Output...")
# fig = plt.figure(figsize=(10, 10))
# plt.scatter(x=word_vectors[:55000,0], y=word_vectors[:55000,1], alpha=1,
#             c=word_frequency[:55000], cmap='rainbow')
# plt.colorbar()
# plt.savefig("bert_normalized.jpg")