from torchtext.vocab import GloVe
import json
import numpy as np
from sklearn.decomposition import PCA
import torch
import torch.nn.functional as F
from tqdm import tqdm
import matplotlib.pyplot as plt

with open("word_frequency.json", "r") as f:
    data = json.load(f)
print("Successful Load Data-File...")

# load the pre-trained word_vectors
vectors = GloVe(cache=".")
print("Successful Load Pre-trained Glove File...Start Mapping Words to Vectors...")
word_vectors = []
word_frequency = []
for d in tqdm(data):
    word_vectors.append(vectors.get_vecs_by_tokens(list(d.keys())[0].strip(), lower_case_backup=True).unsqueeze(0))
    word_frequency.append(list(d.values())[0])
word_vectors = torch.cat(word_vectors, dim=0).numpy()

print("Successful Map Words to Vectors...Start PCA...")
pca = PCA(n_components=2)
word_mean = np.average(word_vectors, axis=0)
pca.fit(word_vectors)

print("Successful Complete PCA...Start Drawing Figures and Output...")
word_vectors = pca.transform(word_vectors)
word_frequency = np.array(word_frequency)
order = np.argsort(word_frequency)
word_frequency = word_frequency[order]
word_vectors = word_vectors[order]
np.save("glove_raw.npy", np.array(word_vectors))
np.save("glove_c.npy", np.array(word_frequency))

# fig = plt.figure(figsize=(10, 10))
# plt.scatter(x=word_vectors[:55000,0], y=word_vectors[:55000,1], alpha=1,
#             c=word_frequency[:55000], cmap='rainbow')
# plt.colorbar()
# plt.savefig("glove_raw.jpg")
# plt.close()

# print("Normalize the Word Vectors by L2 Norm and Repeat the above Steps...")
# word_vectors = []
# word_frequency = []
# for d in tqdm(data):
#     word_vectors.append(vectors.get_vecs_by_tokens(list(d.keys())[0].strip(), lower_case_backup=True).unsqueeze(0))
#     word_frequency.append(list(d.values())[0])
# word_vectors = torch.cat(word_vectors, dim=0)
# word_vectors = F.normalize(word_vectors, dim=-1).numpy()
#
# print("Successful Map Words to Vectors...Start PCA...")
# pca = PCA(n_components=2)
# word_mean = np.average(word_vectors, axis=0)
# pca.fit(word_vectors)
#
# print("Successful Complete PCA...Start Drawing Figures and Output...")
# word_vectors = pca.transform(word_vectors)
# word_frequency = np.array(word_frequency)
# order = np.argsort(word_frequency)
# word_frequency = word_frequency[order]
# word_vectors = word_vectors[order]
# fig = plt.figure(figsize=(10, 10))
# plt.scatter(x=word_vectors[:55000,0], y=word_vectors[:55000,1], alpha=1,
#             c=word_frequency[:55000], cmap='rainbow')
# plt.colorbar()
# plt.savefig("glove_normalized.jpg")