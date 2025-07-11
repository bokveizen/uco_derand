import torch
from torch.autograd import Function
import torch.nn.functional as F
import numpy as np
from torch import softmax
from src.facility_location_methods import *

def sinkhorn_forward(C, mu, nu, epsilon, max_iter):
    bs, n, k_ = C.size()

    v = torch.ones([bs, 1, k_]) / (k_)
    G = torch.exp(-C / epsilon)
    if torch.cuda.is_available():
        v = v.cuda()

    for i in range(max_iter):
        u = mu / (G * v).sum(-1, keepdim=True)
        v = nu / (G * u).sum(-2, keepdim=True)

    Gamma = u * G * v
    return Gamma


def sinkhorn_forward_stablized(C, mu, nu, epsilon, max_iter):
    bs, n, k_ = C.size()
    k = k_ - 1

    f = torch.zeros([bs, n, 1])
    g = torch.zeros([bs, 1, k + 1])
    if torch.cuda.is_available():
        f = f.cuda()
        g = g.cuda()

    epsilon_log_mu = epsilon * torch.log(mu)
    epsilon_log_nu = epsilon * torch.log(nu)

    def min_epsilon_row(Z, epsilon):
        return -epsilon * torch.logsumexp((-Z) / epsilon, -1, keepdim=True)

    def min_epsilon_col(Z, epsilon):
        return -epsilon * torch.logsumexp((-Z) / epsilon, -2, keepdim=True)

    for i in range(max_iter):
        f = min_epsilon_row(C - g, epsilon) + epsilon_log_mu
        g = min_epsilon_col(C - f, epsilon) + epsilon_log_nu

    Gamma = torch.exp((-C + f + g) / epsilon)
    return Gamma


def sinkhorn_backward(grad_output_Gamma, Gamma, mu, nu, epsilon):

    nu_ = nu[:, :, :-1]
    Gamma_ = Gamma[:, :, :-1]

    bs, n, k_ = Gamma.size()

    inv_mu = 1.0 / (mu.view([1, -1]))  # [1, n]
    Kappa = torch.diag_embed(nu_.squeeze(-2)) - torch.matmul(
        Gamma_.transpose(-1, -2) * inv_mu.unsqueeze(-2), Gamma_
    )  # [bs, k, k]

    inv_Kappa = torch.inverse(Kappa)  # [bs, k, k]

    Gamma_mu = inv_mu.unsqueeze(-1) * Gamma_
    L = Gamma_mu.matmul(inv_Kappa)  # [bs, n, k]
    G1 = grad_output_Gamma * Gamma  # [bs, n, k+1]

    g1 = G1.sum(-1)
    G21 = (g1 * inv_mu).unsqueeze(-1) * Gamma  # [bs, n, k+1]
    g1_L = g1.unsqueeze(-2).matmul(L)  # [bs, 1, k]
    G22 = (
        g1_L.matmul(Gamma_mu.transpose(-1, -2)).transpose(-1, -2) * Gamma
    )  # [bs, n, k+1]
    G23 = -F.pad(g1_L, pad=(0, 1), mode="constant", value=0) * Gamma  # [bs, n, k+1]
    G2 = G21 + G22 + G23  # [bs, n, k+1]

    del g1, G21, G22, G23, Gamma_mu

    g2 = G1.sum(-2).unsqueeze(-1)  # [bs, k+1, 1]
    g2 = g2[:, :-1, :]  # [bs, k, 1]
    G31 = -L.matmul(g2) * Gamma  # [bs, n, k+1]
    G32 = (
        F.pad(
            inv_Kappa.matmul(g2).transpose(-1, -2), pad=(0, 1), mode="constant", value=0
        )
        * Gamma
    )  # [bs, n, k+1]
    G3 = G31 + G32  # [bs, n, k+1]

    grad_C = (-G1 + G2 + G3) / epsilon  # [bs, n, k+1]
    return grad_C


class TopKFunc(Function):
    @staticmethod
    def forward(ctx, C, mu, nu, epsilon, max_iter):

        with torch.no_grad():
            if epsilon > 1e-2:
                Gamma = sinkhorn_forward(C, mu, nu, epsilon, max_iter)
                if bool(torch.any(Gamma != Gamma)):
                    print("Nan appeared in Gamma, re-computing...")
                    Gamma = sinkhorn_forward_stablized(C, mu, nu, epsilon, max_iter)
            else:
                Gamma = sinkhorn_forward_stablized(C, mu, nu, epsilon, max_iter)                
            ctx.save_for_backward(mu, nu, Gamma)
            ctx.epsilon = epsilon
        return Gamma

    @staticmethod
    def backward(ctx, grad_output_Gamma):

        epsilon = ctx.epsilon
        mu, nu, Gamma = ctx.saved_tensors
        # mu [1, n, 1]
        # nu [1, 1, k+1]
        # Gamma [bs, n, k+1]
        with torch.no_grad():
            grad_C = sinkhorn_backward(grad_output_Gamma, Gamma, mu, nu, epsilon)
        return grad_C, None, None, None, None


class TopK_custom(torch.nn.Module):
    def __init__(self, k, epsilon=0.1, max_iter=200):
        super(TopK_custom, self).__init__()
        self.k = k
        self.epsilon = epsilon
        self.anchors = torch.FloatTensor([k - i for i in range(k + 1)]).view(
            [1, 1, k + 1]
        )
        self.max_iter = max_iter

        if torch.cuda.is_available():
            self.anchors = self.anchors.cuda()

    def forward(self, scores):
        bs, n = scores.size()
        scores = scores.view([bs, n, 1])

        # find the -inf value and replace it with the minimum value except -inf
        scores_ = scores.clone().detach()
        max_scores = torch.max(scores_).detach()
        scores_[scores_ == float("-inf")] = float("inf")
        min_scores = torch.min(scores_).detach()
        filled_value = min_scores - (max_scores - min_scores)
        mask = scores == float("-inf")
        scores = scores.masked_fill(mask, filled_value)

        C = (scores - self.anchors) ** 2
        C = C / (C.max().detach())

        mu = torch.ones([1, n, 1], requires_grad=False) / n
        nu = [1.0 / n for _ in range(self.k)]
        nu.append((n - self.k) / n)
        nu = torch.FloatTensor(nu).view([1, 1, self.k + 1])

        if torch.cuda.is_available():
            mu = mu.cuda()
            nu = nu.cuda()

        Gamma = TopKFunc.apply(C, mu, nu, self.epsilon, self.max_iter)

        A = Gamma[:, :, : self.k] * n

        return A, None

# https://github.com/ermongroup/subsets/blob/master/subsets/knn/sorting_operator.py

EPSILON = np.finfo(np.float32).tiny

class SubsetOperator(torch.nn.Module):
    def __init__(self, k, tau=1.0, hard=False):
        super(SubsetOperator, self).__init__()
        self.k = k
        self.hard = hard
        self.tau = tau

    def forward(self, scores):
        m = torch.distributions.gumbel.Gumbel(torch.zeros_like(scores), torch.ones_like(scores))
        g = m.sample()
        scores = scores + g

        # continuous top k
        khot = torch.zeros_like(scores)
        onehot_approx = torch.zeros_like(scores)
        for i in range(self.k):
            khot_mask = torch.max(1.0 - onehot_approx, torch.tensor([EPSILON]).cuda())
            scores = scores + torch.log(khot_mask)
            onehot_approx = torch.nn.functional.softmax(scores / self.tau, dim=1)
            khot = khot + onehot_approx

        if self.hard:
            # straight through
            khot_hard = torch.zeros_like(khot)
            val, ind = torch.topk(khot, self.k, dim=1)
            khot_hard = khot_hard.scatter_(1, ind, 1)
            res = khot_hard - khot.detach() + khot
        else:
            res = khot

        return res