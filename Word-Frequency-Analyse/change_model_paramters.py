from transformers import BertForMaskedLM
import torch
import torch.nn.functional as F

m = BertForMaskedLM.from_pretrained("bert-base-uncased")
state_dict = m.state_dict()
decoder_weight = state_dict["cls.predictions.decoder.weight"]
decoder_weight = F.normalize(decoder_weight, dim=1)
state_dict["cls.predictions.decoder.weight"] = decoder_weight
decoder_bias = state_dict["cls.predictions.decoder.bias"]