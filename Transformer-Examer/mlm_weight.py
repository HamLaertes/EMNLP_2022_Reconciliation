import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
from transformers import AutoModelForMaskedLM

m = AutoModelForMaskedLM.from_pretrained("bert-base-uncased")
params = [m.cls.predictions.decoder.weight]

norms = []
for param in params:
    norms += torch.linalg.norm(param, dim=1).tolist()
norms = np.array(norms)

print(np.min(norms), np.max(norms), np.std(norms) / np.mean(norms))
plt.figure(figsize=(7, 3))
plt.hist(norms, bins=40, density=True, stacked=True, facecolor='blue', edgecolor='black')
plt.xlabel("l2-norm values", fontsize=20, fontweight="bold")
plt.ylabel("Ratio", fontsize=20, fontweight="bold")
plt.savefig('weight_norm BERT.pdf', bbox_inches='tight')

# m = AutoModelForMaskedLM.from_pretrained("roberta-base")
# params = [m.lm_head.decoder.weight]
#
# norms = []
# for param in params:
#     norms += torch.linalg.norm(param, dim=1).tolist()
# norms = np.array(norms)
#
# print(np.min(norms), np.max(norms), np.std(norms) / np.mean(norms))
# font = {'size'   : 15}
# matplotlib.rc('font', **font)
# plt.figure(figsize=(8, 5))
# plt.hist(norms, bins=40, density=True, stacked=True, facecolor='blue', edgecolor='black')
# plt.savefig('weight_norm RoBerta.jpg')