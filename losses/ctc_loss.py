import torch
from torch import nn
import numpy as np
import multiprocessing
from torch.autograd import Variable

from configs.loss_config import cfg as loss_cfg


class CTCLoss(nn.Module):
    def __init__(self):
        # self.multiprocessing_pool = multiprocessing.Pool()  # кол-во процессов

        super().__init__()
        self.device = torch.device('cpu')

    def forward(self, out_padded, label, unpadded_w):  # придет b_size выходов нейронноой сети, реальные длины изображений
        # вход: params для каждого эл-та в батче, пслдвти, закодированные в номер в алфавите,
        # реальные длины каждого эл-та в батче деленные на 4
        # out_padded = torch.transpose(out_padded, 0, 1)

        # для каждого i-го эл-та в цикле формируем tuple () всех входов
        # params_to_parallel = [(out, l, w) for out, l, w in zip(out_padded, label, w_unpadded)]
        # подаем в multiprocessing map
        # out = self.multiprocessing_pool(ctc_loss, params_to_parallel)  # list of loss grad
        out = [ctc_loss(out, l, w_unpad) for out, l, w_unpad in zip(out_padded, label, unpadded_w)]
        self.grad = [torch.transpose(out_[1], 1, 0) for out_ in out]
        losses = [out_[0] for out_ in out]
        loss = torch.sum(torch.tensor(losses, requires_grad=True))
        return loss

    def backward(self):
        return Variable(torch.stack(self.grad)).to(self.device), None


def compute_alpha(out_unpadded, label, T, L, bl):  # для альфа и бета
    # m, n = out_unpadded.size()
    # T, L, bl = 0, 0, 0

    a = torch.zeros((T, L), dtype=torch.float32)
    a[0][0] = out_unpadded[0][bl]
    a[0][1] = out_unpadded[0][label[0]]
    c = a[0][0] + a[0][1]
    if c > 0:
        a[0][0] = a[0][0] / c
        a[0][1] = a[0][1] / c
    for t in range(1, T):
        start = torch.max(torch.tensor([0, L - 2 * (T - t)])).item()
        end = torch.min(torch.tensor([2 * t + 2, L])).item()
        for s in range(start, L):
            i = np.max([torch.floor(torch.tensor((s - 1) / 2)).to(dtype=torch.int).item(), 0])
            a[t][s] = a[t - 1][s]
            if s > 0:
                a[t][s] = a[t][s] + a[t-1][s-1]
            if s % 2 == 0:
                a[t][s] = a[t][s] * out_unpadded[t][bl]
            elif s == 1 or label[i] == label[i - 1]:
                a[t][s] = a[t][s] * out_unpadded[t][label[i]]
            else:
                a[t][s] = (a[t][s] + a[t - 1][s - 2]) * out_unpadded[t][label[i]]
        c = torch.sum(a[t][start: end])
        if c > 0:
            for s in range(start, end):
                a[t][s] = a[t][s] / c
    return a


def ctc_loss(out_padded, label, unpadded_w):
    out_padded = torch.transpose(out_padded, 1, 0)
    M, N = out_padded.size()
    w = unpadded_w.item() // 4
    bl = 0

    L = 2 * len(label) + 1
    T = w
    out_unpadded = torch.zeros((T, N))
    for t in range(0, T):
        for n in range(0, N):
            out_unpadded[t][n] = out_padded[t][n]
    a = compute_alpha(out_unpadded, label, T, L, bl)
    out_unpadded_flipped = torch.fliplr(out_unpadded)

    idx = [i for i in range(label.size(0) - 1, -1, -1)]
    idx = torch.LongTensor(idx)
    label_reversed = label.index_select(0, idx)

    b = compute_alpha(out_unpadded_flipped, label_reversed, T, L, bl)
    b = torch.flipud(torch.fliplr(b))
    ab = a * b
    lab = torch.zeros((T, N))
    for s in range(0, L):
        if s % 2 == 0:
            for t in range(T):
                lab[t][bl] = lab[t][bl] + ab[t][s]
                ab[t][s] = ab[t][s] / out_unpadded[t][bl]
        else:
            for t in range(T):
                i = np.max([torch.floor(torch.tensor((s-1) / 2)).to(dtype=torch.int).item(), 0])
                lab[t][label[i]] = lab[t][label[i]] + ab[t][s]
                ab[t][s] = ab[t][s] / out_unpadded[t][label[i]]

    lh = torch.zeros(T)
    for t in range(T):
        lh[t] = torch.sum(ab[t])
    # print(f'lh: {lh}')
    loss = - torch.sum(torch.log(lh))
    softmax_grad = torch.zeros((M, N))
    for t in range(T):
        for n in range(N):
            softmax_grad[t][n] = out_unpadded[t][n] - lab[t][n] / (out_unpadded[t][n] * lh[t])
            if torch.isnan(softmax_grad[t][n]):
                softmax_grad[t][n] = 0
    return loss, softmax_grad


if __name__ == '__main__':
    ctc_loss = CTCLoss()
