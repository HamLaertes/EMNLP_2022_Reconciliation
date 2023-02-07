import numpy as np
from transformers import BertTokenizer
from tqdm import tqdm
import json

def get_sentence(filelines):
    filelines = [line.split('\t') for line in filelines]
    words, tags = zip(*filelines)
    words = [word.lower() for word in words]
    return words, tags

samples = []
tags = []
with open('data/inter/train.txt', 'r', encoding='utf-8') as f:
    lines = f.readlines()
samplelines = []
for line in lines:
    line = line.strip()
    if line:
        samplelines.append(line)
    else:
        sample, sample_tag = get_sentence(samplelines)
        samples.append(sample)
        tags.append(sample_tag)
        samplelines = []

vocab_count = np.load("vocab_count_2500000.npy")

entity_vocab_count = {}
entity_vocab = {}
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
print(len(samples))
for i, (words, tags) in enumerate(tqdm(zip(samples, tags))):
    # if i == 10:
    #     break
    for word, tag in zip(words, tags):
        token_ids = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(word))
        if len(token_ids) == 0:
            continue
        token_id = int(token_ids[0])
        tmp = entity_vocab_count.get(tag, [])
        tmp.append(int(vocab_count[token_id]))
        entity_vocab_count[tag] = tmp
        tmp = entity_vocab.get(tag, [])
        tmp.append(token_id)
        entity_vocab[tag] = tmp

# for k, v in entity_vocab_count.items():
#     print(k, type(k))
#     print(v, type(v), v[0], type(v[0]))

with open("entity_vocab_count.json", "w") as f:
    json.dump(entity_vocab_count, f)
entity_vocab_count = [v for v in entity_vocab_count.values()]
entity_vocab_count = [sum(v) / len(v) for v in entity_vocab_count]
entity_vocab_count = np.array(entity_vocab_count)
np.save("entity_vocab_count.npy", entity_vocab_count)
with open("entity_vocab.json", "w") as f:
    json.dump(entity_vocab, f)
