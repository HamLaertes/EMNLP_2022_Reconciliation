import matplotlib.pyplot as plt
import numpy as np

glove_raw = np.load("glove_raw.npy")
glove_c = np.load("glove_c.npy")
bert_raw = np.load("bert_raw.npy")
bert_c = np.load("bert_c.npy")
roberta_raw = np.load("roberta_raw.npy")
roberta_c = np.load("roberta_c.npy")

tick_size = 14
fig, ax = plt.subplots(1, 2, figsize=(10, 4))
print(ax.shape)
# ax = fig.add_subplot(131)

# ax.set_title("GloVe", fontsize=14)
ax[0].set_title("BERT", fontsize=25, fontweight="bold")
ax[1].set_title("RoBERTa", fontsize=24, fontweight="bold")

# plt.tick_params(weight="bold")
# labels = ax.get_xticklabels() + ax.get_yticklabels() + \
#          bx.get_xticklabels() + bx.get_yticklabels() + \
#          cx.get_xticklabels() + cx.get_yticklabels()
# [label.set_fontsize(tick_size) for label in labels]

# a = ax.scatter(x=glove_raw[:55000,0], y=glove_raw[:55000,1], alpha=1,
#              c=glove_c[:55000], cmap='rainbow')
b = ax[0].scatter(x=bert_raw[:55000,0], y=bert_raw[:55000,1], alpha=1,
             c=bert_c[:55000], cmap='rainbow', rasterized=True)
c = ax[1].scatter(x=roberta_raw[:55000,0], y=roberta_raw[:55000,1], alpha=1,
             c=roberta_c[:55000], cmap='rainbow', rasterized=True)
colorbar = fig.colorbar(b, ax=ax)

plt.savefig("freq_emb.pdf", bbox_inches='tight')