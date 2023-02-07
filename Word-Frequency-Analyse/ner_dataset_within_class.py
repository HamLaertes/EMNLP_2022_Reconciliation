import matplotlib.pyplot as plt
import numpy as np
import json

with open("entity_vocab_count.json") as f:
    e_v = json.load(f)

e1, e2 = np.random.randint(1, len(e_v.keys()), (2,))
e_1 = sorted(e_v[list(e_v.keys())[e1]])[1500:-1500]
e_2 = sorted(e_v[list(e_v.keys())[e2]])[1500:-1500]

fig = plt.figure(figsize=(16, 4))
ax = fig.add_subplot(121)
bx = fig.add_subplot(122)

# a = 0.3
# b = 1
# k = (b - a) / (np.max(e_1) - np.min(e_1))
# e_1 = a + k * (e_1 - np.min(e_1))
# k = (b - a) / (np.max(e_2) - np.min(e_2))
# e_2 = a + k * (e_2 - np.min(e_2))

ax.hist(e_1, bins=50, density=True, facecolor='red', edgecolor='black', alpha=0.8)
ax.set_xticks([])
ax.set_yticks([])
ax.set_ylabel("Ratio", fontweight="bold", fontsize=30)
ax.set_xlabel("Word Frequencies", fontweight="bold", fontsize=30)
bx.hist(e_2, bins=50, density=True, facecolor='red', edgecolor='black', alpha=0.8)
bx.set_xticks([])
bx.set_yticks([])
bx.set_ylabel("Ratio", fontweight="bold", fontsize=30)
bx.set_xlabel("Word Frequencies", fontweight="bold", fontsize=30)
# plt.suptitle("The word frequencies in two entities.", fontsize=25, fontweight="bold")
plt.savefig('wf_w.pdf', bbox_inches='tight')
