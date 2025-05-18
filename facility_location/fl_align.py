from src.facility_location_methods import *
from src.facility_location_data import *
from src.poi_bin import pmf_poibin
from rich import traceback
from torch.nn import functional as F

traceback.install()

def incremental_differences(probs, dist, k, egn_beta, s_max=100, small_card_th=2, small_card_ratio=10):
    # alignment: soft derandomization of probs
    n_nodes = probs.shape[0]
    if s_max <= 0:
        s_max = n_nodes + 1
    probs_matrix = probs.view(1, -1)
    k_diff_remove = torch.abs(
        torch.arange(n_nodes, device=probs_matrix.device) - k
    )  # [n]
    if small_card_th > 0:
        k_diff_remove[:small_card_th] *= small_card_ratio    
    k_diff_add = torch.abs(
        torch.arange(n_nodes, device=probs_matrix.device) + 1 - k
    )  # [n]
    if small_card_th > 1:
        k_diff_add[:small_card_th - 1] *= small_card_ratio    

    card_dist_orig = pmf_poibin(
        probs_matrix, probs_matrix.device, use_normalization=False
    )
    dist_sort = torch.sort(dist, dim=1)
    dist_ordering = dist_sort.indices
    dist_ordered = dist_sort.values
    
    p_reordered = probs_matrix[:, dist_ordering]
    q_reordered = 1 - p_reordered

    q_reordered_cumprod = q_reordered.cumprod(dim=-1).roll(
        shifts=1, dims=-1
    )
    q_reordered_cumprod[..., 0] = 1.0
    p_closest = q_reordered_cumprod * p_reordered
    
    # p2, constriant 2
    xx_ = probs_matrix
    pmf_cur = card_dist_orig
    # [0, 0.5]
    term_1a = (1.0 / (1 - xx_)).unsqueeze(-1)
    # (1 - X_v)^{-1}; [b, n, 1]; term_1a[ib, iv] = (1 - X^{(b)}_v)^{-1}
    q_flip = pmf_cur.flip(1)
    q_roll_stack = torch.tril(
        torch.as_strided(
            q_flip.repeat(1, 2),
            (q_flip.shape[0], q_flip.shape[1], q_flip.shape[1]),
            (q_flip.shape[1] * 2, 1, 1),
        ).flip(1)
        # )[:, :n_nodes]
    )[:, :n_nodes, :s_max]
    # q_roll_stack[ib, i] = (q_i, q_{i-1}, ..., q_0, 0, ..., 0), 0 <= i < n; [b, n, n + 1]
    term_2a = (xx_ / (xx_ - 1.0)).unsqueeze(-1) ** torch.arange(
        # n_nodes + 1, device=probs_matrix.device
        s_max,
        device=probs_matrix.device,
    )
    # term_2a[ib, iv, i] = (X^{(b)}_v / (X^{(b)}_v - 1))^{i}; [b, n, n + 1]
    # term_2a = torch.einsum("bix, bvx -> bvi", q_roll_stack, term_2a)  # [b, n, n]
    term_2a = term_2a @ q_roll_stack.transpose(1, 2)  # [b, n, n]

    res_case_1 = term_1a * term_2a
    # del term_1a, term_2a

    # [0.5, 1]
    term_1b = (1.0 / xx_).unsqueeze(-1)
    q = pmf_cur
    q_roll_stack = torch.tril(
        torch.as_strided(
            q.repeat(1, 2),
            (q.shape[0], q.shape[1], q.shape[1]),
            (q.shape[1] * 2, 1, 1),
        ).flip(1)
        # )[:, :n_nodes].flip(1)
    )[:, :n_nodes].flip(1)[..., :s_max]
    # q_roll_stack[ib, i] = (q_{i + 1}, q_{i + 2}, ..., q_n, 0, ..., 0), 0 <= i < n; [b, n, n + 1]
    term_2b = ((xx_ - 1.0) / xx_).unsqueeze(-1) ** torch.arange(
        # n_nodes + 1, device=probs_matrix.device
        s_max,
        device=probs_matrix.device,
    )
    # term_2b = torch.einsum("bix, bvx -> bvi", q_roll_stack, term_2b)
    term_2b = term_2b @ q_roll_stack.transpose(1, 2)  # [b, n, n]

    res_case_2 = term_1b * term_2b
    # del term_1b, term_2b

    tilde_q = torch.where(xx_.unsqueeze(-1) <= 0.5, res_case_1, res_case_2)
    tilde_q.clamp_(0.0, 1.0)

    pmf_new = tilde_q[..., :n_nodes]  # [b, n, n]
    dol_remove_p2 = (pmf_new * k_diff_remove).sum(dim=-1)  # [b, n]
    dol_add_p2 = (pmf_new * k_diff_add).sum(dim=-1)  # [b, n]

    vx2result = torch.empty(
        probs_matrix.shape[0], probs_matrix.shape[1], 2
    ).to(probs_matrix.device)
    vx2result[..., 0] = dol_remove_p2 * egn_beta
    vx2result[..., 1] = dol_add_p2 * egn_beta

    obj_expand = dist_ordered * p_closest  # [b, n, n]; (b, v, i) -> p_i d_i
    p_closest_cumsum = (
        p_closest.flip(-1).cumsum(-1).flip(-1).roll(-1, dims=-1)
    )
    p_closest_cumsum[..., -1] = 0  # [n, n]; (v, i) -> \sum_{j > i} p_j
    pd_cumsum = obj_expand.flip(-1).cumsum(-1).flip(-1).roll(-1, dims=-1)
    pd_cumsum[..., -1] = 0  # [n, n]; (v, i) -> \sum_{j > i} p_j d_j
    # p_ratio = p_reordered / (1 - p_reordered)  # [n, n]; (v, i) -> X_{ui} / (1 - X_{ui})
    p_ratio = probs_matrix / (
        1 - probs_matrix
    )  # [n]; (u) -> X_u / (1 - X_u)
    inv_ordering = dist_ordering.argsort(dim=-1).expand(
        1, -1, -1
    )
    # fv_diff_add(u) = \sum_v p_closest_cumsum[v, inv_ordering[v, u]] dist[v, u] - pd_cumsum[v, inv_ordering[v, u]]
    fv_diff_add = (
        torch.gather(p_closest_cumsum, dim=-1, index=inv_ordering) * dist
        - torch.gather(pd_cumsum, dim=-1, index=inv_ordering)
    ).sum(dim=-2)
    # fv_diff_remove(u) = \sum_v pd_cumsum[v, inv_ordering[v, u]] * p_ratio[v, inv_ordering[v, u]] - p_cloest[v, inv_ordering[v, u]] * dist[v, u]
    # fv_diff_remove(u) = \sum_v pd_cumsum[v, inv_ordering[v, u]] * p_ratio[u] - p_cloest[v, inv_ordering[v, u]] * dist[v, u]
    fv_diff_remove = (
        torch.gather(pd_cumsum, dim=-1, index=inv_ordering)
        * p_ratio.unsqueeze(-2)
        - torch.gather(p_closest, dim=-1, index=inv_ordering) * dist
    ).sum(dim=-2)
    vx2result[..., 0] += fv_diff_remove
    vx2result[..., 1] += fv_diff_add

    vx2result = vx2result.squeeze(0)  # [n_nodes, 2]
    
    # for facility location, small is better
    return -vx2result

def alignment_together(probs, vx2result, softmax_func, sign_func, shift):
    vx2softmax = softmax_func(vx2result)
    step0, step1 = vx2softmax[:, 0], vx2softmax[:, 1]
    step_diff = step1 - step0
    sign_right = sign_func(step_diff)
    
    if shift:
        probs = probs + step_diff * (probs + sign_right * (1 - 2 * probs))
        probs = probs.clamp(0.0, 1.0)
    else:
        probs = vx2softmax[:, 1]
    
    return probs

def softmax_func_normal(x, temp):
    softmax_flatten = F.softmax(x.flatten() / temp, dim=-1)
    return softmax_flatten.view_as(x)


def softmax_func_gumbel(x, temp, hard=False):
    softmax_flatten = F.gumbel_softmax(x.flatten(), tau=temp, hard=hard, dim=-1)
    return softmax_flatten.view_as(x)

def alignment_greedy(probs, vx2result, softmax_func1, softmax_func2, sign_func):        
    vx2softmax = softmax_func1(vx2result)  # [n_nodes, 2]; row-wise sum = 1
    step0, step1 = vx2softmax[:, 0], vx2softmax[:, 1]  # [n_nodes]
    step_diff = step1 - step0  # [n_nodes]
    sign_right = sign_func(step_diff)  # [n_nodes]
    
    sx2softmax_value = (vx2softmax * vx2result).sum(dim=-1)  # [n_nodes]
    choose_node_softmax = softmax_func2(sx2softmax_value)  # [n_nodes]; sum = 1
    step_diff = step_diff * choose_node_softmax  # [n_nodes]
           
    probs = probs + step_diff * (probs + sign_right * (1 - 2 * probs)) 
    probs = probs.clamp(0.0, 1.0)
    return probs

def alignment_greedy_flatten(probs, vx2result, softmax_func, sign_func):
    vx2softmax = softmax_func(vx2result)
    step0, step1 = vx2softmax[:, 0], vx2softmax[:, 1]
    step_diff = step1 - step0  # [n_nodes]
    sign_right = sign_func(step_diff)  # [n_nodes]
    probs = probs + step_diff * (probs + sign_right * (1 - 2 * probs))
            
    return probs

def alignment_greedy_flatten_single_node(probs, i_node, vx2result_i, softmax_func, sign_func):
    vx2softmax = softmax_func(vx2result_i)
    step0, step1 = vx2softmax[0], vx2softmax[1]        
    step_diff = step1 - step0  # [1]
    sign_right = sign_func(step_diff)  # [1]
    probs_i_new = probs[i_node] + step_diff * (probs[i_node] + sign_right * (1 - 2 * probs[i_node]))        
    probs_new = probs.clone()
    probs_new[i_node] = probs_i_new
    
    return probs_new

def alignment_single_node_given_val(probs, i_node, best_val, vx2result_i, softmax_func):
    vx2softmax = softmax_func(vx2result_i)
    step_i = vx2softmax[best_val]
    probs_i_new = probs[i_node] + step_i * (best_val - probs[i_node])    
    probs_new = probs.clone()
    probs_new[i_node] = probs_i_new        
    return probs_new
