from datasets import load_dataset
from transformers import BertTokenizer, BertForMaskedLM
import random
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

dataset = load_dataset("bookcorpus", split='train[0:10000]')
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

def tokenize(example):
    token_ids = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(example['text']))
    token_ids = token_ids[:126]
    len_ = len(token_ids)
    mask_ind = random.randint(0, len_-1)
    example['labels'] = token_ids[mask_ind]
    example['mask_inds'] = mask_ind + 1
    token_ids[mask_ind] = tokenizer.mask_token_id
    token_ids = [tokenizer.cls_token_id] + token_ids + [tokenizer.sep_token_id]
    attention_mask = [1 for _ in range(len(token_ids))]
    token_ids_pad = token_ids + [tokenizer.pad_token_id for _ in range(128-len(token_ids))]
    assert len(token_ids_pad) == 128
    attention_mask += [0 for _ in range(128-len(token_ids))]
    assert len(attention_mask) == 128
    example['input_ids'] = token_ids_pad
    example['attention_mask'] = attention_mask
    return example

def collate_fn(data):
    batch = {'input_ids': [], "attention_mask": [], "labels": [], "mask_inds": []}
    for i in range(len(data)):
        for k in batch:
            batch[k].append(torch.tensor(data[i][k], dtype=torch.long))
    for k in batch:
        batch[k] = torch.stack(batch[k], 0)
    return batch

dataset = dataset.map(tokenize)
dataloader = DataLoader(dataset, batch_size=32, num_workers=4, collate_fn=collate_fn)

loss_fct = nn.CrossEntropyLoss()
model = BertForMaskedLM.from_pretrained("bert-base-uncased").cuda()
model.eval()
state_dict = model.state_dict()
#state_dict["cls.predictions.decoder.bias"].fill_(0)
#state_dict["cls.predictions.transform.dense.bias"].fill_(0)
#state_dict["cls.predictions.bias"].fill_(0)
model.load_state_dict(state_dict)
total = 0
acc = 0
loss = []
for batch in tqdm(dataloader):
    for k in batch:
        batch[k] = batch[k].cuda()
    labels = batch.pop("labels")
    mask_inds = batch.pop("mask_inds")
    outputs = model.bert(**batch)
    w_rep = outputs[0]
    w_rep = model.cls.predictions.transform(w_rep)
    raw_predict = model.cls.predictions.decoder(w_rep)
    predict = []
    for i in range(raw_predict.size()[0]):
        predict.append(raw_predict[i][mask_inds[i]])
    predict = torch.stack(predict, 0)
    loss_ = loss_fct(predict, labels)
    predict = torch.argmax(predict, dim=1)
    acc_ = (predict == labels).sum()
    acc += acc_.item()
    total += labels.size()[0]
    loss.append(loss_.item())
print(acc / total, sum(loss) / len(loss))

dataloader = DataLoader(dataset, batch_size=32, num_workers=4, collate_fn=collate_fn)
total = 0
acc = 0
loss = []
for batch in tqdm(dataloader):
    for k in batch:
        batch[k] = batch[k].cuda()
    labels = batch.pop("labels")
    mask_inds = batch.pop("mask_inds")
    outputs = model.bert(**batch)
    w_rep = outputs[0]
    w_rep = F.normalize(w_rep, dim=-1)
    w_rep = model.cls.predictions.transform(w_rep)
    raw_predict = model.cls.predictions.decoder(w_rep)
    predict = []
    for i in range(raw_predict.size()[0]):
        predict.append(raw_predict[i][mask_inds[i]])
    predict = torch.stack(predict, 0)
    predict = F.normalize(predict, dim=-1)
    loss_ = loss_fct(predict, labels)
    predict = torch.argmax(predict, dim=1)
    acc_ = (predict == labels).sum()
    acc += acc_.item()
    total += labels.size()[0]
    loss.append(loss_.item())
print(acc / total, sum(loss) / len(loss))

state_dict = model.state_dict()
decoder_weight = state_dict["cls.predictions.decoder.weight"]
decoder_weight = F.normalize(decoder_weight, dim=-1)
state_dict["cls.predictions.decoder.weight"] = decoder_weight
# transform_weight = state_dict['cls.predictions.transform.dense.weight']
# transform_weight = F.normalize(transform_weight, dim=-1)
# state_dict['cls.predictions.transform.dense.weight'] = transform_weight
model.load_state_dict(state_dict)

dataloader = DataLoader(dataset, batch_size=32, num_workers=4, collate_fn=collate_fn)
total = 0
acc = 0
loss = []
for batch in tqdm(dataloader):
    for k in batch:
        batch[k] = batch[k].cuda()
    labels = batch.pop("labels")
    mask_inds = batch.pop("mask_inds")
    outputs = model.bert(**batch)
    w_rep = outputs[0]
    w_rep = model.cls.predictions.transform(w_rep)
    raw_predict = model.cls.predictions.decoder(w_rep)
    predict = []
    for i in range(raw_predict.size()[0]):
        predict.append(raw_predict[i][mask_inds[i]])
    predict = torch.stack(predict, 0)
    loss_ = loss_fct(predict, labels)
    predict = torch.argmax(predict, dim=1)
    acc_ = (predict == labels).sum()
    acc += acc_.item()
    total += labels.size()[0]
    loss.append(loss_.item())
print(acc / total, sum(loss) / len(loss))