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
import os

def __load_model__(ckpt):
    '''
    ckpt: Path of the checkpoint
    return: Checkpoint dict
    '''
    if os.path.isfile(ckpt):
        checkpoint = torch.load(ckpt)
        print("Successfully loaded checkpoint '%s'" % ckpt)
        return checkpoint
    else:
        raise Exception("No checkpoint found at '%s'" % ckpt)

with open("word_frequency.json", "r") as f:
    data = json.load(f)
print("Successful Load Data-File...")

# load the pre-trained word_vectors
few_setting = "proto_l2_5_5"
load_ckpt = "../Few-NERD-main/{}.pth.tar".format(few_setting)
bert = BertModel.from_pretrained("bert-base-uncased")
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
state_dict = __load_model__(load_ckpt)['state_dict']
own_state = bert.state_dict()
copied_num = 0
for name, param in state_dict.items():
    if name.startswith("word_encoder.module.bert."):
        name = name[25:]
    if name not in own_state:
        continue
    own_state[name].copy_(param)
    copied_num += 1
bert.load_state_dict(own_state)
if copied_num != len(own_state):
    print("{} number of parameters are not found in ckpt.".format(len(own_state)-copied_num))
bert_embedding = bert.embeddings
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

print("Successful Map Words to Vectors..Start Drawing Figures and Output...")
word_frequency = np.array(word_frequency)
word_vectors = np.linalg.norm(word_vectors, axis=-1)
order = np.argsort(word_frequency)
word_frequency = word_frequency[order]
word_vectors = word_vectors[order]
fig = plt.figure(figsize=(10, 10))
plt.scatter(x=word_vectors[:55000], y=word_frequency[:55000], alpha=1,
            c=word_frequency[:55000], cmap='rainbow')
plt.colorbar()
plt.savefig("bert_norm_frequency_{}.jpg".format(few_setting))
plt.close()