# Normalizing Prototypes in Few-Shot Named Entity Recognition

This is the source code of the *Findings of EMNLP 2023* paper: [**Reconciliation of Pre-trained Models and Prototypical Neural Networks in Few-shot Named Entity Recognition**](https://arxiv.org/abs/2211.03270)

We demonstrate a simple yet effective method to improve the performance of Few-Shot Named Entity Recognition.
The method is plug-and-play. 
Therefore, most of the codes in this repository are copied from the dataset repositories: [MetaWSD](https://github.com/Nithin-Holla/MetaWSD) and [Few-NERD](https://github.com/thunlp/Few-NERD).

In this README.md, we only highlight where our method brings differences in the two dataset codes. 
For detailed introductions, e.g. how to prepare the datasets, how to run the codes and test the models, users can refer to the two dataset repositories.
We exactly follow the same experimental settings.

## Contents

We give a simple introductions to each directory.

- Big-Frequency-Norm: This directory shows how to draw the Figure 2 in our paper. It examines the relationship between words frequencies and the l2-norms of the words embeddings.
- Few-NERD-main: This directory contains the codes relates to the experiments on Few-NERD.
- MetaWSD-master: This directory contains the codes relates to the experiments on MetaWSD.
- Transformer-Examer: This directory shows how to draw the Figure 7 in our paper.
- Word-Frequency-Analyse: This directory shows how to draw the Figures 3-6 in our paper.

## Our method

We highlight where our method brings differences in the two dataset codes.

### MetaWSD

The codes build prototypical neural networks is MetaWSD-master/models/seq_proto.py.
The initial code piece in the dataset repository is in the lines 180-183:
```pycon
d = torch.stack(
            tuple([q.sub(p).pow(2).sum(dim=-1) for p in prototypes]),
            dim=-1
        )
```
We change it (in the lines 224-230) to normalize the prototypes:
```pycon
        if self.norm:
            prototypes = F.normalize(prototypes, dim=-1)
        if not self.dot:
            d = torch.sqrt(torch.stack(
                tuple([q.sub(p).pow(2).sum(dim=-1) for p in prototypes]),
                dim=-1
            )).neg()
```

### Few-NERD

The codes build prototypical neural networks is in Few-NERD-main/model/proto.py
The initial code piece in the dataset repository is in the lines 16-20:
```pycon
    def __dist__(self, x, y, dim):
        if self.dot:
            return (x * y).sum(dim)
        else:
            return -(torch.pow(x - y, 2)).sum(dim)
```
We change it into (in the lines 21-30) to normalize the prototypes:
```pycon
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
```

## Note

We do not clean this repository to have a clear and nice organization because the authors do not have enough times on this.
We are sorry.
As our method is fairly easy to build (only costs one line to normalize the prototypes), we believe the above introductions tells the key information on reproducing our experiments.
If you have any questions, please give an email to the first author.
