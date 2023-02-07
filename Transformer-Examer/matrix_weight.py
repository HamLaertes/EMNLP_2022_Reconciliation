import matplotlib.pyplot as plt

import torch
from transformers import BertModel

m = BertModel.from_pretrained("bert-base-uncased")
params = []
for n, p in m.named_parameters():
    if "key.weight" in n:
        params.append(p.data)

norms = []
for param in params:
    norms += torch.linalg.norm(param, dim=1).tolist()

plt.figure(figsize=(10, 10))
plt.hist(norms, bins=50, density=True, facecolor='blue', edgecolor='black')
plt.savefig('key_norm.jpg')

params = []
for n, p in m.named_parameters():
    if "query.weight" in n:
        params.append(p.data)

norms = []
for param in params:
    norms += torch.linalg.norm(param, dim=1).tolist()

plt.figure(figsize=(10, 10))
plt.hist(norms, bins=50, density=True, facecolor='blue', edgecolor='black')
plt.savefig('query_norm.jpg')

params = []
for n, p in m.named_parameters():
    if "value.weight" in n:
        params.append(p.data)

norms = []
for param in params:
    norms += torch.linalg.norm(param, dim=1).tolist()

plt.figure(figsize=(10, 10))
plt.hist(norms, bins=50, density=True, facecolor='blue', edgecolor='black')
plt.savefig('value_norm.jpg')