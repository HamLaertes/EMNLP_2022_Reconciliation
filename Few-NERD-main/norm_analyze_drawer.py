import matplotlib
import matplotlib.pyplot as plt
import numpy as np

norm_proto = np.load("norm_proto.npy")
print(np.min(norm_proto), np.max(norm_proto), np.std(norm_proto) / np.mean(norm_proto))

max_ = np.load("max_proto_norm.npy")
min_ = np.load("min_proto_norm.npy")
mean_ = np.load("avg_proto_norm.npy")
x = np.load("proto_norm_c.npy")

fig = plt.figure(figsize=(7, 3))
plt.hist(norm_proto, bins=40, density=True, facecolor='red', edgecolor='black', alpha=0.8)
plt.xlabel("l2-norm values", fontweight="bold", fontsize=20)
plt.ylabel("Ratio", fontweight="bold", fontsize=20)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
# plt.xlabel('l2-norm', fontsize=18)
plt.savefig("nerd_norm_proto.pdf", bbox_inches='tight')
plt.close()

def create_multi_bars(labels, datas, tick_step=1, group_gap=0.2, bar_gap=0):
    plt.figure(figsize=(7, 3))
    '''
    labels : x轴坐标标签序列
    datas ：数据集，二维列表，要求列表每个元素的长度必须与labels的长度一致
    tick_step ：默认x轴刻度步长为1，通过tick_step可调整x轴刻度步长。
    group_gap : 柱子组与组之间的间隙，最好为正值，否则组与组之间重叠
    bar_gap ：每组柱子之间的空隙，默认为0，每组柱子紧挨，正值每组柱子之间有间隙，负值每组柱子之间重叠
    '''
    # ticks为x轴刻度
    ticks = np.arange(len(labels)) * tick_step
    # group_num为数据的组数，即每组柱子的柱子个数
    group_num = len(datas)
    # group_width为每组柱子的总宽度，group_gap 为柱子组与组之间的间隙。
    group_width = tick_step - group_gap
    # bar_span为每组柱子之间在x轴上的距离，即柱子宽度和间隙的总和
    bar_span = group_width / group_num
    # bar_width为每个柱子的实际宽度
    bar_width = bar_span - bar_gap
    # baseline_x为每组柱子第一个柱子的基准x轴位置，随后的柱子依次递增bar_span即可
    baseline_x = ticks - (group_width - bar_span) / 2
    for index, y in enumerate(datas):
        plt.bar(baseline_x + index*bar_span, y, bar_width)
    # plt.ylabel('l2-norms')
    # x轴刻度标签位置与x轴刻度一致
    plt.xticks(ticks, labels)
    plt.xlabel("Entity ID", fontweight="bold", fontsize=20)
    plt.ylabel("l2-norm values", fontweight="bold", fontsize=20)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.savefig("class-l2-norm.pdf", bbox_inches='tight')

print(x)
print(len(max_), len(min_), len(mean_))
create_multi_bars(x, [max_, min_, mean_])