import matplotlib.pyplot as plt
import numpy as np
import json

vocab_count = np.load("entity_vocab_count.npy")
with open("entity_vocab.json") as f:
    e_v = json.load(f)

a = 0.3
b = 1
k = (b - a) / (np.max(vocab_count) - np.min(vocab_count))
v_c = a + k * (vocab_count - np.min(vocab_count))
plt.figure(figsize=(16, 4))
plt.xticks([])
plt.yticks([])
plt.bar([i for i in range(vocab_count.shape[0])], vocab_count, color=plt.get_cmap('YlOrBr')(v_c))
plt.xlabel('Entity ID', fontsize=28, fontweight="bold")
plt.ylabel('Frequencies', fontsize=28, fontweight="bold")
# plt.title('The mean words frequencies of different entities.', fontsize=25, fontweight="bold")
plt.savefig('wf_e.pdf', bbox_inches='tight')

# o = set(e_v.pop('O'))
# mo_rc = []
# for k, v in e_v.items():
#     mo_rc.append(len(set(v)))
#
# mo_c = []
# for k, v in e_v.items():
#     mo_c.append(len(set(v) - o))
# print(mo_rc)
# print(mo_c)

# vocab_count = np.load("vocab_count_2500000.npy")
# for k, vs in e_v.items():
#     tmp = []
#     for v in vs:
#         tmp.append(vocab_count[v])
#     e_v[k] = sum(tmp) / len(tmp)
#
# e_v = sorted(e_v.items(), key=lambda x: x[1], reverse=True)
# print(e_v)
