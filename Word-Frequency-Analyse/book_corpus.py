from datasets import load_dataset
from transformers import BertTokenizer
import numpy as np
import matplotlib.pyplot as plt

dataset = load_dataset("bookcorpus", split='train[0:2500000]')
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

vocab_count = [0 for _ in range(tokenizer.vocab_size)]
def tokenize(example):
    token_ids = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(example['text']))
    for token_id  in token_ids:
        vocab_count[token_id] += 1
dataset.map(tokenize)
vocab_count = np.array(vocab_count)
np.save("vocab_count_2500000.npy", vocab_count)
