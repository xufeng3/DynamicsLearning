import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import copy
import random
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.modules import Linear
import torch.nn.functional as F
import argparse
import time
from tqdm import tqdm
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

N = 1000
p = 0.05
infect_init = 100
gamma = 0.04
beta = 0.08
T = 100


def args_init():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=777)
    parser.add_argument('--batch_size', type=int, default=5)
    parser.add_argument('--lr', type=float, default=0.001,
                        help='learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.001,
                        help='weight decay')
    parser.add_argument('--nhid', type=int, default=128,
                        help='hidden size')
    parser.add_argument('--pooling_ratio', type=float, default=0.5,
                        help='pooling ratio')
    parser.add_argument('--dropout_ratio', type=float, default=0.5,
                        help='dropout ratio')
    parser.add_argument('--epochs', type=int, default=30,
                        help='maximum number of epochs')
    parser.add_argument('--patience', type=int, default=50,
                        help='patience for earlystopping')
    parser.add_argument('--visdom', type=bool, default='False',
                        help='use visdom or not')
    parser.add_argument('--device', type=str, default='cpu')
    args = parser.parse_args()
    return args


def alpha(l):
    return 1-(1-gamma)**l


G = nx.random_graphs.erdos_renyi_graph(N, p)
s0 = [random.randint(0, 1) for _ in range(N)] # random init
state = []
state.append(s0)
# 0 for S and 1 for I
sample_data = []
for _ in range(T):
    current_state = []
    last_state = state[-1]
    sample_idx = np.random.choice(N, N//10, replace=False).tolist()
    for i in range(N):
        last = last_state[i]
        current = last
        neighbors = list(G.neighbors(i))
        l = 0
        neighbor_state = []
        for neighbor in neighbors:
            l += last_state[neighbor]
            neighbor_state.append(last_state[neighbor])
        p = alpha(l) if last == 0 else beta
        r = random.random()
        if r<p:
            current = 1 if last == 0 else 0
        current_state.append(current)
        # sample
        if i in sample_idx:
            y = [1-p, p] if last == 0 else [beta, 1-beta]
            data = y + [last] + [l] + neighbor_state
            sample_data.append(data)
    state.append(current_state)


# ground truth plot
plt.subplots()
l = []
y1 = []
y2 = []
# pred_y1 = np.array(pred_y)[:, 0].tolist()
# pred_y2 = np.array(pred_y)[:, 1].tolist()

for i in range(len(sample_data)):
    l.append(sample_data[i][3])
    y1.append(sample_data[i][0])
    y2.append(sample_data[i][1])
plt.scatter(l, y1)
plt.scatter(l, y2)
# plt.scatter(l, pred_y1)
# plt.scatter(l, pred_y2)
plt.show()


# class MyData(Dataset):
#     def __init__(self, data):
#         self.data = data
#         super(MyData, self).__init__()
#
#     def __getitem__(self, idx):
#         return self.data[idx]
#
#     def __len__(self):
#         return len(self.data)
#
#
#
# class Net(torch.nn.Module):
#     def __init__(self):
#         super(Net, self).__init__()
#         self.lin = Linear(1, 32)
#         self.att_a = Linear(32, 1)
#         self.att_b = Linear(32, 1)
#         self.lin1 = Linear(32, 2)
#
#     def forward(self, ln, lm, lx, cx):
#         ln = ln.view(-1, 1)
#         lx = lx.view(-1, 1)
#         cx = cx.view(-1, 1)
#         cx1 = F.relu(self.lin(cx))
#         ln1 = F.relu(self.lin(ln))
#         ln2 = F.relu(self.att_b(ln1))
#         lx1 = F.relu(self.lin(lx))
#         lx2 = F.relu(self.att_a(lx1))
#         att = F.sigmoid(ln2 + lx2).view(1, -1)
#         xj0 = att @ ln1 + lx1
#         xj = self.mask_sum(xj0, lm)
#         xj = xj.T
#         res = F.relu(self.lin1(xj + cx1))
#         return res
#
#     def mask_sum(self, x, mask):
#         list = []
#         for i in range(int(torch.max(mask))+1):
#             s = x[mask == i, :].sum(dim=0).view(-1, 1)
#             list.append(s)
#         res = torch.cat(list, dim=1)
#         return res
#
#
# def test(loader):
#     model.eval()
#     correct = 0.
#     loss = 0.
#     pred_y = []
#     for data in loader:
#         ln, lm, lx, cx, y = data
#         ln = ln.to(args.device)
#         lm = lm.to(args.device)
#         lx = lx.to(args.device)
#         cx = cx.to(args.device)
#         y = y.to(args.device)
#         out = model(ln, lm, lx, cx)
#         loss += F.mse_loss(out, y, reduction='sum').item()
#         pred_y.append(out.detach().cpu().numpy().tolist()[0])
#     print(pred_y)
#     return pred_y, loss/len(loader.dataset)
#
#
# def collate_fn(batch):
#     ln, lm, lx, cx, y = [], [], [], [], []
#     for i, item in enumerate(batch):
#         yy, x, l, neighbor = item[0:2], item[1], item[3], item[4:]
#         lm += [i] * len(neighbor)
#         ln += neighbor
#         lx += [last] * len(neighbor)
#         cx += [last]
#         y.append(yy)
#     ln = torch.Tensor(ln)
#     lm = torch.Tensor(lm)
#     lx = torch.Tensor(lx)
#     cx = torch.Tensor(cx)
#     y = torch.Tensor(y)
#     return ln, lm, lx, cx, y
#
#
# # 0: init and load data
# args = args_init()
# args.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
# data_train = MyData(sample_data)
# data_val = MyData(sample_data)
# # data_test = MyData(sample_data)
# loader_train = DataLoader(data_train, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
# loader_val = DataLoader(data_val, batch_size=1, shuffle=False, collate_fn=collate_fn)
# # loader_test = DataLoader(data_test, batch_size=args.batch_size, shuffle=False)
# model = Net().to(args.device)
# optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
#
# # 1: train
# min_loss = 1e10
# patience = 0
# for epoch in range(args.epochs):
#     model.train()
#     for data in loader_train:
#         ln, lm, lx, cx, y = data
#         ln = ln.to(args.device)
#         lm = lm.to(args.device)
#         lx = lx.to(args.device)
#         cx = cx.to(args.device)
#         y = y.to(args.device)
#         out = model(ln, lm, lx, cx)
#         loss = F.mse_loss(out, y)
#         print("Training loss:{}".format(loss.item()))
#         loss.backward()
#         optimizer.step()
#         optimizer.zero_grad()
#     pred_y, loss_val = test(loader_val)
#     # patience machanism
#     if loss_val < min_loss:
#         dtw = time.strftime("%Y%m%d%H%M", time.localtime(int(time.time())))  # 获取当前时间
#         torch.save(model.state_dict(), 'latest0.pth')
#         print("Model saved at epoch{}".format(epoch))
#         min_loss = loss_val
#         patience = 0
#     else:
#         patience += 1
#     if patience > args.patience:
#         break
#
# # # 2: test
# # model = Net().to(args.device)
# # model.load_state_dict(torch.load('latest0.pth'))
# # loss_test = test(loader_test)
# # print(loss_test)
#
#
# # gt plot
# plt.subplots()
# l = []
# y1 = []
# y2 = []
# pred_y1 = np.array(pred_y)[:, 0].tolist()
# pred_y2 = np.array(pred_y)[:, 1].tolist()
#
# for i in range(len(sample_data)):
#     l.append(sample_data[i][3])
#     y1.append(sample_data[i][0])
#     y2.append(sample_data[i][1])
# # plt.scatter(l, y1)
# # plt.scatter(l, y2)
# plt.scatter(l, pred_y1)
# plt.scatter(l, pred_y2)
# plt.show()
