import numpy as np
import matplotlib.pyplot as plt

lw=2.3
fontsize=18
fontsize_legend=10
fig = plt.figure(figsize=(15, 5))
ax = fig.add_subplot(121)
bx = fig.add_subplot(122)
ax.set_title("INTER 5 way 5~10 shot", fontsize=fontsize)
bx.set_title("INTRA 10 way 1~2 shot", fontsize=fontsize)

plt.subplots_adjust(wspace=0.5)

train_x = np.arange(1, 101)
dev_x = np.arange(10, 101, 10)
norm_train = []
norm_dev = []
with open("proto_l2_norm_inter_5_5_.txt", "r") as f:
    for line in f.readlines():
        line = line.strip('\n')
        if "Finish" in line:
            break
        if "step" not in line:
            continue
        # print(line)
        f1_score = float(line.split("f1:")[1].strip())
        if "EVAL" in line:
            norm_dev.append(float(f1_score))
        else:
            norm_train.append(float(f1_score))

origin_train_1e_5 = []
origin_dev_1e_5 = []
with open("proto_l2_inter_5_5.txt", "r") as f:
    for line in f.readlines():
        line = line.strip('\n')
        if "Finish" in line:
            break
        if "step" not in line:
            continue
        f1_score = float(line.split("f1:")[1].strip())
        if "EVAL" in line:
            origin_dev_1e_5.append(float(f1_score))
        else:
            origin_train_1e_5.append(float(f1_score))

origin_train_1e_4 = []
origin_dev_1e_4 = []
with open("proto_l2_inter_5_5_lr1e-4.txt", "r") as f:
    for line in f.readlines():
        line = line.strip('\n')
        if "Finish" in line:
            break
        if "step" not in line:
            continue
        f1_score = float(line.split("f1:")[1].strip())
        if "EVAL" in line:
            origin_dev_1e_4.append(float(f1_score))
        else:
            origin_train_1e_4.append(float(f1_score))

ax.plot(train_x, norm_train, c='red', lw=lw, label='Ours Train')
ax.plot(dev_x, norm_dev, c='red', linestyle='--', marker='*', lw=lw, label='Ours Dev')
ax.plot(train_x, origin_train_1e_5, c='blue', lw=lw, label='Original Train*')
ax.plot(dev_x, origin_dev_1e_5, c='blue', marker='*', linestyle='--', lw=lw, label='Original Train*')
ax.plot(train_x, origin_train_1e_4, c='green', lw=lw, label='Original Train')
ax.plot(dev_x, origin_dev_1e_4, c='green', marker='*', linestyle='--', lw=lw, label='Original Train')
ax.tick_params(axis='both', which='major', labelsize=fontsize)
ax.set_xlabel("epochs", fontsize=fontsize)
ax.set_ylabel("f1 scores", fontsize=fontsize)
ax.legend(loc = "best", fontsize=fontsize_legend)#图例

norm_train = []
norm_dev = []
with open("proto_l2_norm_intra_10_1.txt", "r") as f:
    for line in f.readlines():
        line = line.strip('\n')
        if "Finish" in line:
            break
        if "step" not in line:
            continue
        # print(line)
        f1_score = float(line.split("f1:")[1].strip())
        if "EVAL" in line:
            norm_dev.append(float(f1_score))
        else:
            norm_train.append(float(f1_score))

origin_train_1e_5 = []
origin_dev_1e_5 = []
with open("proto_l2_intra_10_1.txt", "r") as f:
    for line in f.readlines():
        line = line.strip('\n')
        if "Finish" in line:
            break
        if "step" not in line:
            continue
        f1_score = float(line.split("f1:")[1].strip())
        if "EVAL" in line:
            origin_dev_1e_5.append(float(f1_score))
        else:
            origin_train_1e_5.append(float(f1_score))

origin_train_1e_4 = []
origin_dev_1e_4 = []
with open("proto_l2_intra_10_1_lr1e-4.txt", "r") as f:
    for line in f.readlines():
        line = line.strip('\n')
        if "Finish" in line:
            break
        if "step" not in line:
            continue
        f1_score = float(line.split("f1:")[1].strip())
        if "EVAL" in line:
            origin_dev_1e_4.append(float(f1_score))
        else:
            origin_train_1e_4.append(float(f1_score))

bx.plot(train_x, norm_train, c='red', lw=lw, label='Ours Train')
bx.plot(dev_x, norm_dev, c='red', linestyle='--', marker='*', lw=lw, label='Ours Dev')
bx.plot(train_x, origin_train_1e_5, c='blue', lw=lw, label='Original Train*')
bx.plot(dev_x, origin_dev_1e_5, c='blue', marker='*', linestyle='--', lw=lw, label='Original Train*')
bx.plot(train_x, origin_train_1e_4, c='green', lw=lw, label='Original Train')
bx.plot(dev_x, origin_dev_1e_4, c='green', marker='*', linestyle='--', lw=lw, label='Original Train')
bx.set_xlabel("epochs", fontsize=fontsize)
bx.set_ylabel("f1 scores", fontsize=fontsize)
bx.tick_params(axis='both', which='major', labelsize=fontsize)
bx.legend(loc = "best", fontsize=fontsize_legend)#图例
plt.savefig("learning_draw.pdf", bbox_inches='tight')