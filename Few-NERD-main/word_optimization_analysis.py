import torch
import torch.utils.data as data
import os
import numpy as np
import random
import argparse
from util.word_encoder import BERTWordEncoder
from tqdm import tqdm
import json
import torch.nn.functional as F
from transformers import AutoTokenizer
import matplotlib.pyplot as plt

class TrainDataset(data.Dataset):
    """
    Fewshot NER Dataset
    """

    def __init__(self, filepaths, tokenizer, max_length):
        for filepath in filepaths:
            if not os.path.exists(filepath):
                print("[ERROR] Data file does not exist!")
                assert (0)
        self.tokenizer = tokenizer
        self.samples = self.__load_data_from_file__(filepaths)
        self.max_length = max_length

    def __load_data_from_file__(self, filepaths):
        samples = []
        for filepath in filepaths:
            with open(filepath, 'r', encoding='utf-8')as f:
                lines = f.readlines()
            samplelines = []
            for line in lines:
                line = line.strip()
                if line:
                    samplelines.append(line)
                else:
                    filelines = [line.split('\t') for line in samplelines]
                    words, tags = zip(*filelines)
                    words = [word.lower() for word in words]
                    samples.append(words)
                    samplelines = []
        return samples

    def __get_token_list__(self, sample):
        tokens = []
        for word in sample:
            word_tokens = self.tokenizer.tokenize(word)
            if word_tokens:
                tokens.extend(word_tokens)
        return tokens

    def __getraw__(self, tokens):
        # get tokenized word list, attention mask, text mask (mask [CLS], [SEP] as well)

        # split into chunks of length (max_length-2)
        # 2 is for special tokens [CLS] and [SEP]
        tokens_list = []
        while len(tokens) > self.max_length - 2:
            tokens_list.append(tokens[:self.max_length - 2])
            tokens = tokens[self.max_length - 2:]
        if tokens:
            tokens_list.append(tokens)

        # add special tokens and get masks
        indexed_tokens_list = []
        mask_list = []
        text_mask_list = []
        for i, tokens in enumerate(tokens_list):
            # token -> ids
            tokens = ['[CLS]'] + tokens + ['[SEP]']
            indexed_tokens = self.tokenizer.convert_tokens_to_ids(tokens)

            # padding
            while len(indexed_tokens) < self.max_length:
                indexed_tokens.append(0)
            indexed_tokens_list.append(indexed_tokens)

            # mask
            mask = np.zeros((self.max_length), dtype=np.int32)
            mask[:len(tokens)] = 1
            mask_list.append(mask)

            # text mask, also mask [CLS] and [SEP]
            text_mask = np.zeros((self.max_length), dtype=np.int32)
            text_mask[1:len(tokens) - 1] = 1
            text_mask_list.append(text_mask)

        return indexed_tokens_list, mask_list, text_mask_list

    def __additem__(self, d, word, mask, text_mask):
        d['word'] += word
        d['mask'] += mask
        d['text_mask'] += text_mask

    def __populate__(self, sample):
        '''
        populate samples into data dict
        set savelabeldic=True if you want to save label2tag dict
        'index': sample_index
        'word': tokenized word ids
        'mask': attention mask in BERT
        'label': NER labels
        'sentence_num': number of sentences in this set (a batch contains multiple sets)
        'text_mask': 0 for special tokens and paddings, 1 for real text
        '''
        dataset = {'word': [], 'mask': [], 'text_mask': []}
        tokens = self.__get_token_list__(sample)
        word, mask, text_mask = self.__getraw__(tokens)
        word = torch.tensor(word).long()
        mask = torch.tensor(mask).long()
        text_mask = torch.tensor(text_mask).long()
        self.__additem__(dataset, word, mask, text_mask)
        return dataset

    def __getitem__(self, index):
        sample = self.samples[index]
        sample = self.__populate__(sample)
        return sample

    def __len__(self):
        return len(self.samples)

def collate_fn(data):
    batch_data = {'word': [], 'mask': [], 'text_mask': []}
    for i in range(len(data)):
        for k in batch_data:
            batch_data[k] += data[i][k]
    for k in batch_data:
        batch_data[k] = torch.stack(batch_data[k], 0)
    return batch_data


def get_loader(filepath, tokenizer, batch_size, max_length, num_workers=8, collate_fn=collate_fn):
    dataset = TrainDataset(filepath, tokenizer, max_length)
    data_loader = data.DataLoader(dataset=dataset,
                                  batch_size=batch_size,
                                  shuffle=False,
                                  pin_memory=True,
                                  num_workers=num_workers,
                                  collate_fn=collate_fn)
    return data_loader

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

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

@torch.no_grad()
def get_emb(model, data):
    return model(data['word'].cuda(), data['mask'].cuda())

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', default=4, type=int,
                        help='batch size')
    parser.add_argument('--max_length', default=32, type=int,
                        help='max length')
    parser.add_argument('--load_ckpt', default=None,
                        help='load ckpt')
    parser.add_argument('--fp16', action='store_true',
                        help='use nvidia apex fp16')
    parser.add_argument('--seed', type=int, default=0,
                        help='random seed')

    opt = parser.parse_args()
    batch_size = opt.batch_size
    max_length = opt.max_length

    print("max_length: {}, batch_size: {}".format(max_length, batch_size))

    set_seed(opt.seed)
    print('loading tokenizer and dataset...')
    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
    data_files = ["./data/inter/train.txt"]
    data_loader = get_loader(data_files, tokenizer, batch_size, max_length)
    print('loading model...')
    word_encoder = BERTWordEncoder("bert-base-uncased")

    word_encoder.eval()
    word_encoder.cuda()
    raw_word_emb = []
    for batch_idx, data in tqdm(enumerate(data_loader)):
        word_emb = get_emb(word_encoder, data)
        valid_word_emb = word_emb.view(-1, word_emb.size()[-1])[data['text_mask'].view(-1)==1]
        raw_word_emb.append(valid_word_emb.cpu())
    raw_word_emb = torch.cat(raw_word_emb, dim=0)

    print('load trained model...')
    few_setting = "proto_cos_5_5"
    load_ckpt = "{}.pth.tar".format(few_setting)
    state_dict = __load_model__(load_ckpt)['state_dict']
    own_state = word_encoder.state_dict()
    copied_num = 0
    for name, param in state_dict.items():
        if name.startswith("word_encoder.module."):
            name = name[20:]
        if name not in own_state:
            continue
        own_state[name].copy_(param)
        copied_num += 1
    word_encoder.load_state_dict(own_state)
    if copied_num != len(own_state):
        print("{} number of parameters are not found in ckpt.".format(len(own_state) - copied_num))

    word_encoder.eval()
    word_encoder.cuda()
    trained_word_emb = []
    for batch_idx, data in tqdm(enumerate(data_loader)):
        word_emb = get_emb(word_encoder, data)
        valid_word_emb = word_emb.view(-1, word_emb.size()[-1])[data['text_mask'].view(-1) == 1]
        trained_word_emb.append(valid_word_emb.cpu())
    trained_word_emb = torch.cat(trained_word_emb, dim=0)

    norm_diff = (torch.linalg.norm(raw_word_emb, dim=-1) - torch.linalg.norm(trained_word_emb, dim=-1)).numpy()
    norm_relative_diff = torch.linalg.norm(raw_word_emb - trained_word_emb, dim=-1).numpy()
    angle_diff = 1 - torch.matmul(F.normalize(raw_word_emb, dim=-1).unsqueeze(1), F.normalize(trained_word_emb, dim=-1).unsqueeze(-1)).squeeze().numpy()
    print(norm_diff.shape, angle_diff.shape)
    fig = plt.figure(figsize=(20, 10))
    plt.hist(norm_diff, bins=50, density=True, facecolor='blue', edgecolor='black', alpha=1)
    plt.savefig('norm_diff_{}.jpg'.format(few_setting))
    plt.close()
    fig = plt.figure(figsize=(20, 10))
    plt.hist(norm_relative_diff, bins=50, density=True, facecolor='red', edgecolor='black', alpha=1)
    plt.savefig('norm_relative_diff_{}.jpg'.format(few_setting))
    plt.close()
    fig = plt.figure(figsize=(20, 10))
    plt.hist(angle_diff, bins=50, density=True, color='green', edgecolor='black', alpha=1)
    plt.savefig('angle_diff_{}.jpg'.format(few_setting))
    plt.close()

if __name__ == "__main__":
    main()