import multiprocessing
import torch
import torch.nn as nn
import numpy as np

from torch.autograd import Variable

cuda = torch.cuda.is_available()
device = "cpu"  # torch.device("cuda:0" if cuda else "cpu")
print(device)

eps = np.finfo(float).eps


def decode_best_path(probs_list, blank=0):
    seq_list = []
    for probs in probs_list:
        best_path = np.argmax(probs, axis=0).tolist()
        seq = []
        for i, b in enumerate(best_path):
            if b == blank:
                continue
            elif i != 0 and b == best_path[i - 1]:
                continue
            else:
                seq.append(b)
        seq_list.append(seq)
    return seq_list


def compute_alpha(params, seq, T, L, blank):
    alphas = np.zeros((L, T))
    alphas[0, 0] = params[blank, 0]
    alphas[1, 0] = params[seq[0], 0]
    c = np.sum(alphas[:, 0])
    if c > 0:
        alphas[:, 0] = alphas[:, 0] / c
    for t in range(1, T):
        start = max(0, L - 2 * (T - t))
        end = min(2 * t + 2, L)
        for s in range(start, L):
            l = max(0, (s - 1) // 2)
            # blank
            if s % 2 == 0:
                if s == 0:
                    alphas[s, t] = alphas[s, t - 1] * params[blank, t]
                else:
                    alphas[s, t] = (alphas[s, t - 1] + alphas[s - 1, t - 1]) * params[blank, t]
            # same label twice
            elif s == 1 or seq[l] == seq[l - 1]:
                alphas[s, t] = (alphas[s, t - 1] + alphas[s - 1, t - 1]) * params[seq[l], t]
            else:
                alphas[s, t] = (alphas[s, t - 1] + alphas[s - 1, t - 1] + alphas[s - 2, t - 1]) \
                               * params[seq[l], t]

        c = np.sum(alphas[start:end, t])
        if c > 0:
            alphas[start:end, t] = alphas[start:end, t] / c
    return alphas


def ctc_loss(args):
    params, seq, length, blank = args
    ch, t = params.shape
    final_grad = np.zeros((ch, t))
    if t == 0 or length == 0:
        return 0.0, final_grad
    seqLen = len(seq)
    L = 2 * seqLen + 1
    T = length
    params = params[:, :T]
    lab_absum = np.zeros(params.shape)
    a = compute_alpha(params, seq, T, L, blank)
    b_flipped = compute_alpha(np.fliplr(params), seq[::-1], T, L, blank)
    b = np.flipud(np.fliplr(b_flipped))

    # b = np.zeros((L, T))
    # b[-1, T - 1] = params[blank, T - 1]
    # b[-2, T - 1] = params[seq[-1], T - 1]
    # c = np.sum(b[:, T - 1])
    # if c > 0:
    #     b[:, T - 1] = b[:, T - 1] / c
    # for t in range(T - 2, -1, -1):
    #     start = max(0, L - 2 * (T - t))
    #     end = min(2 * t + 2, L)
    #     for s in range(end - 1, -1, -1):
    #         l = max(0, (s - 1) // 2)
    #         # blank
    #         if s % 2 == 0:
    #             if s == L - 1:
    #                 b[s, t] = b[s, t + 1] * params[blank, t]
    #             else:
    #                 b[s, t] = (b[s, t + 1] + b[s + 1, t + 1]) * params[blank, t]
    #         # same label twice
    #         elif s == L - 2 or seq[l] == seq[l + 1]:
    #             b[s, t] = (b[s, t + 1] + b[s + 1, t + 1]) * params[seq[l], t]
    #         else:
    #             b[s, t] = (b[s, t + 1] + b[s + 1, t + 1] + b[s + 2, t + 1]) \
    #                           * params[seq[l], t]
    #
    #     c = np.sum(b[start:end, t])
    #     if c > 0:
    #         b[start:end, t] = b[start:end, t] / c

    ab = a * b

    for s in range(L):
        # blank
        if s % 2 == 0:
            lab_absum[blank, :] += ab[s, :]
            ab[s, :] = ab[s, :] / (params[blank, :])
        else:
            l = max(0, (s - 1) // 2)
            lab_absum[seq[l], :] += ab[s, :]
            ab[s, :] = ab[s, :] / (params[seq[l], :])
    lh = np.sum(ab, 0)
    loss = -np.sum(np.log(lh))
    final_grad[:, :T] = params - lab_absum / (params * lh)
    nan_indices = np.isnan(final_grad)
    # print(str(len(np.where(nan_indices)[0])) + '/' + str(final_grad.shape[0] * final_grad.shape[1]))
    final_grad[nan_indices] = 0.0

    return loss, final_grad


class CTCLoss(nn.Module):
    def __init__(self):
        super(CTCLoss, self).__init__()
        # self.pool = multiprocessing.Pool(1)

    def forward(self, params_list, seq_list, lengths, blank=0):
        dynamic_params = [(params_list[i].numpy(), np.asarray(seq_list[i]), np.asarray(lengths[i]) // 4, blank) for i in
                          range(len(params_list))]
        result = [ctc_loss(params) for params in dynamic_params]  # self.pool.map(ctc_loss, dynamic_params)
        self.grad = [result[i][1] for i in range(len(result))]
        losses = [result[i][0] for i in range(len(result))]
        loss_sum = np.sum(losses)
        return loss_sum

    def backward(self):
        return (Variable(torch.Tensor(self.grad)).to(device), None)
