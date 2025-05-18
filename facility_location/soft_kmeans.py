import torch
from torch import softmax

def soft_kmeans(dist, probs, topk_func, dist_tau=1.0):
    centers = topk_func(probs)
    membership_softmax = softmax(-dist / dist_tau, dim=-1)    
    # soft assignment: assignment[i, j] = Pr[i is in the cluster with center j]
    assignment = membership_softmax * centers.unsqueeze(0)
    assignment = assignment / assignment.sum(dim=1, keepdim=True)
    
    # within each cluster, compute the expected total distance
    #   expected_distance[i, j] 
    # = E[total distance from i to all points in the cluster with center j]
    # = sum_k assignment[k, j] * dist[i, k]
    expected_distance = dist @ assignment

    # (soft) choosing new centers
    new_centers = softmax(-expected_distance / dist_tau, dim=0)
    new_centers = new_centers * centers.unsqueeze(0)
    new_centers = new_centers.sum(-1)
    
    return new_centers