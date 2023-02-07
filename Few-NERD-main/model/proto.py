import sys
sys.path.append('..')
import util
import torch
from torch import autograd, optim, nn
from torch.autograd import Variable
from torch.nn import functional as F
import math

class Proto(util.framework.FewShotNERModel):
    
    def __init__(self, word_encoder, dot=False, ignore_index=-1, n_way=5, k_shot=5):
        util.framework.FewShotNERModel.__init__(self, word_encoder, ignore_index=ignore_index)
        self.drop = nn.Dropout()
        self.protos_bias = nn.Sequential(
            nn.Linear(n_way+1, word_encoder.bert.config.hidden_size),
            nn.Softmax(dim=-1),
        )
        self.dot = dot

    def __dist__(self, x, y, dim):
        if self.dot:
            x = F.normalize(x, dim=-1).squeeze(0)
            y = y.squeeze(1)
            # y = F.normalize(y, dim=-1).squeeze(1)
            return torch.matmul(y, x.transpose(0, 1))
        else:
            x = F.normalize(x, dim=-1).squeeze(0)
            # y = F.normalize(y, dim=-1).squeeze(0)
            return -(torch.pow(x - y, 2)).sum(dim)

    def __batch_dist__(self, S, Q, q_mask):
        # S [class, embed_dim], Q [num_of_sent, num_of_tokens, embed_dim]
        assert Q.size()[:2] == q_mask.size()
        Q = Q[q_mask==1].view(-1, Q.size(-1)) # [num_of_all_text_tokens, embed_dim]
        return self.__dist__(S.unsqueeze(0), Q.unsqueeze(1), 2)

    def norm_diff(self, S, Q, q_mask, label):
        norm_diff = []
        Q = Q[q_mask == 1].view(-1, Q.size(-1))
        label = torch.cat(label, 0)
        for l in range(torch.max(label)+1):
            q = Q[label==l]
            s = S[l].unsqueeze(0)
            tmp = torch.linalg.norm(q, dim=-1) - torch.linalg.norm(s, dim=-1)
            norm_diff += tmp.tolist()
        return norm_diff

    def __get_proto__(self, embedding, tag, mask):
        proto = []
        embedding = embedding[mask==1].view(-1, embedding.size(-1))
        tag = torch.cat(tag, 0)
        assert tag.size(0) == embedding.size(0)
        for label in range(torch.max(tag)+1):
            proto.append(torch.mean(embedding[tag==label], 0))
        proto = torch.stack(proto)
        return proto

    def forward(self, support, query):
        '''
        support: Inputs of the support set.
        query: Inputs of the query set.
        N: Num of classes
        K: Num of instances for each class in the support set
        Q: Num of instances in the query set
        '''
        support_emb = self.word_encoder(support['word'], support['mask'], (None, None, None))
        query_emb = self.word_encoder(query['word'], query['mask'], (None, None, None))

        support_emb = self.drop(support_emb)
        query_emb = self.drop(query_emb)

        # Prototypical Networks
        logits = []
        current_support_num = 0
        current_query_num = 0
        assert support_emb.size()[:2] == support['mask'].size()
        assert query_emb.size()[:2] == query['mask'].size()

        for i, sent_support_num in enumerate(support['sentence_num']):
            sent_query_num = query['sentence_num'][i]
            # Calculate prototype for each class
            support_proto = self.__get_proto__(
                support_emb[current_support_num:current_support_num+sent_support_num],
                support['label'][current_support_num:current_support_num+sent_support_num], 
                support['text_mask'][current_support_num: current_support_num+sent_support_num])
            # calculate distance to each prototype
            logits.append(self.__batch_dist__(
                support_proto,
                query_emb[current_query_num:current_query_num+sent_query_num],
                query['text_mask'][current_query_num: current_query_num+sent_query_num])) # [num_of_query_tokens, class_num]
            current_query_num += sent_query_num
            current_support_num += sent_support_num
        logits = torch.cat(logits, 0)
        _, pred = torch.max(logits, 1)

        # protos_for_log = torch.cat(protos_for_log, dim=0)
        # query_emb_for_log = torch.cat(query_emb_for_log, dim=0)
        # sigma = torch.ones_like(protos_for_log, device=protos_for_log.device)
        # prior_dist = torch.distributions.normal.Normal(protos_for_log, sigma)
        # log_p = prior_dist.log_prob(query_emb_for_log).sum(-1)

        return logits, pred, None

    def norm_analyze(self, support, query, label):
        support_emb = self.word_encoder(support['word'], support['mask'], protos_bias=(None, None, None))
        query_emb = self.word_encoder(query['word'], query['mask'], protos_bias=(None, None, None))

        protos = []
        protos_dict = {}
        current_support_num = 0
        for i, sent_support_num in enumerate(support['sentence_num']):
            sent_query_num = query['sentence_num'][i]
            label2tag = query['label2tag'][i]
            # Calculate prototype for each class
            support_proto = self.__get_proto__(
                support_emb[current_support_num:current_support_num + sent_support_num],
                support['label'][current_support_num:current_support_num + sent_support_num],
                support['text_mask'][current_support_num: current_support_num + sent_support_num])
            for k, v in label2tag.items():
                tmp = protos_dict.get(v, [])
                tmp.append(support_proto[k].cpu())
                protos_dict[v] = tmp
            protos.append(support_proto.unsqueeze(0).expand(sent_query_num, -1, -1))
            current_support_num += sent_support_num
        protos = torch.cat(protos, dim=0)

        text_mask = []
        current_query_num = 0
        for i, sent_query_num in enumerate(query['sentence_num']):
            text_mask.append(query['text_mask'][current_query_num: current_query_num + sent_query_num])
            current_query_num += sent_query_num

        norm_diff = []
        current_support_num = 0
        current_query_num = 0
        for i, sent_support_num in enumerate(support['sentence_num']):
            sent_query_num = query['sentence_num'][i]
            # Calculate prototype for each class
            support_proto = self.__get_proto__(
                support_emb[current_support_num:current_support_num+sent_support_num],
                support['label'][current_support_num:current_support_num+sent_support_num],
                support['text_mask'][current_support_num: current_support_num+sent_support_num])
            # calculate distance to each prototype
            norm_diff += (self.norm_diff(
                support_proto,
                query_emb[current_query_num:current_query_num+sent_query_num],
                query['text_mask'][current_query_num: current_query_num+sent_query_num],
                query['label'][current_query_num: current_query_num + sent_query_num])) # [num_of_query_tokens, class_num]
            current_query_num += sent_query_num
            current_support_num += sent_support_num
        norm_proto = torch.linalg.norm(protos.view(-1, protos.size()[-1]), dim=-1).detach().cpu()

        return norm_diff, norm_proto, protos_dict
