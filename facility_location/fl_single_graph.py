from pathlib import Path
from src.facility_location_methods import *
from fl_align import *
from src.facility_location_data import *
from src.poi_bin import pmf_poibin_vec
from rich import traceback
from torch.nn import functional as F
import argparse
from soft_topk_ot import *
from soft_kmeans import *

traceback.install()

####################################
#             config               #
####################################

parser = argparse.ArgumentParser()

parser.add_argument("--n_nodes", type=int, default=100)
parser.add_argument("--n_choose", type=int, default=10)
parser.add_argument("--i_data", type=int, default=0)
parser.add_argument("--i_init", type=int, default=0)

parser.add_argument("--nep", type=int, default=100)
parser.add_argument("--lr", type=float, default=0.01)
parser.add_argument("--reg", type=float, default=0.01)
parser.add_argument("--timestamp", type=str, default="")

# additional arguments for the "aligned" version
parser.add_argument("--warmup_ep", type=int, default=0)  # number of warmup epochs without alignment

# softmax temperature
parser.add_argument("--temp", type=float, default=1.0)  # softmax temperature
# temperature scheduling: temp = temp * {tempdecay} every {tempfreq} epochs
parser.add_argument("--temp_decay", type=float, default=1.0)
parser.add_argument("--temp_freq", type=int, default=1)

# Gumbel softmax
parser.add_argument("--gumbel", action="store_true")  # use gumbel softmax or not
parser.add_argument("--gumbel_onehot", action="store_true")  # gumbel uses onehot (straight-through) or not

parser.add_argument("--regratio", type=float, default=1.0)
# regratio scheduling: regratio = regratio * {regratiodecay} every {regratiofreq} epochs
parser.add_argument("--regratio_decay", type=float, default=1.0)
parser.add_argument("--regratio_freq", type=int, default=1)

parser.add_argument("--sign_sigmoid", type=float, default=-1)  # sigmoid scaling for sign (<0 = no sigmoid)

parser.add_argument("--no_align", action="store_true")  # no alignment
parser.add_argument("--seq_align", action="store_true")  # using sequential alignment
parser.add_argument("--simultaneous", action="store_true")  # using simultaneous alignment or not
parser.add_argument("--align_iter", type=int, default=1)  # number of alignment iterations

parser.add_argument("--smax", type=int, default=10)  # s_max
parser.add_argument("--sct", type=int, default=0)  # small_card_th
parser.add_argument("--scr", type=int, default=10)  # small_card_ratio

# topk
parser.add_argument("--no_topk", action="store_true")  # use topk or not
parser.add_argument("--topk_type", type=str, default="ot")  # topk type (ot or gumbel; others = no topk)
parser.add_argument("--topk_eps", type=float, default=1.0)  # epsilon for topk
parser.add_argument("--topk_iter", type=int, default=200)  # max iteration for topk

# kmeans
parser.add_argument("--no_kmeans", action="store_true")  # use kmeans or not
parser.add_argument("--kmeans_temp", type=float, default=1.0)  # temperature for kmeans
parser.add_argument("--kmeans_iter", type=int, default=1)  # max iteration for kmeans

args = parser.parse_args()

device = torch.device("cuda:0")

####################################
#            training              #
####################################

try:
    data = torch.load(
        f"data/facility_location_rand{args.n_nodes}_{args.i_data}.pt",
        map_location=device,
    )
    points = data
except:
    # train_dataset = get_random_data(args.n_nodes, 2, 0, device)
    # _, data = train_dataset[0]
    data = torch.rand(args.n_nodes, 2, device=device)
    points = data
    torch.save(data.cpu(), f"data/facility_location_rand{args.n_nodes}_{args.i_data}.pt")

from time import strftime, localtime

def get_local_time():
    return strftime("%Y%m%d%H%M%S", localtime())


def param2string(param_):
    return str(param_).replace(".", "p")


lr_val = args.lr
lr_string = param2string(lr_val)
reg_val = args.reg
reg_string = param2string(reg_val)

if args.timestamp:
    timestamp = args.timestamp
else:
    timestamp = get_local_time()

softmax_temp = args.temp
softmax_decay = args.temp_decay
softmax_freq = args.temp_freq

regratio = args.regratio
regratio_decay = args.regratio_decay
regratio_freq = args.regratio_freq

EPS = 1e-3

to_align = not args.no_align
to_soft_kmeans = not args.no_kmeans
to_soft_topk = not args.no_topk

graph, dist = build_graph_from_points(data, None, True, "euclidean")

n_points = args.n_nodes
i_init = args.i_init

try:
    probs_logits = torch.load(f"probs_logits_{n_points}_{i_init}.pt", map_location=device)
except:
    probs_logits = torch.randn(n_points, device=device)
    torch.save(probs_logits, f"probs_logits_{n_points}_{i_init}.pt")

probs_logits = torch.nn.Parameter(probs_logits)

optimizer = torch.optim.Adam([probs_logits], lr=lr_val)

s_max = args.smax
small_card_th = args.sct
small_card_ratio = args.scr

k = args.n_choose

# softmax function
if not args.gumbel:  # using normal softmax
    if args.simultaneous:
        softmax_func = lambda x: torch.softmax(x / softmax_temp, dim=-1)
    else:  # not simultaneous (flattened)
        softmax_func = lambda x: softmax_func_normal(x, softmax_temp)
else:  # using Gumbel softmax
    if args.simultaneous:
        softmax_func = lambda x: F.gumbel_softmax(
            x, tau=softmax_temp, hard=args.gbonehot, dim=-1
        )
    else:  # not simultaneous (flattened)
        softmax_func = lambda x: softmax_func_gumbel(x, softmax_temp, args.gbonehot)

# sign function
if args.sign_sigmoid <= 0:  # not using sigmoid for sign
    sign_func = lambda x: (x > 0).float()  # 1 if x > 0, 0 otherwise
else:
    sign_func = lambda x: torch.sigmoid(args.sign_sigmoid * x)

# top-k and kmeans
soft_topk_func_ot = TopK_custom(k, epsilon=args.topk_eps, max_iter=args.topk_iter)
soft_topk_func_gumbel = SubsetOperator(k=k, tau=args.topk_eps, hard=False)

def soft_topk_func_(probs_input, type="ot"):
    if type == "ot":
        ps2, _ = soft_topk_func_ot(torch.log(probs_input.unsqueeze(0)))
        return ps2.sum(dim=2).squeeze(0)
    # gumbel
    probs_input = probs_input.unsqueeze(0)
    return soft_topk_func_gumbel(probs_input).squeeze(0)

# sum-k probs --> new sum-k probs

soft_topk_func = lambda x: soft_topk_func_(x, type=args.topk_type)
soft_kmeans_func = lambda x: soft_kmeans(dist, x, soft_topk_func, dist_tau=args.kmeans_temp)

n_epochs = args.nep

for epoch in trange(n_epochs):
    probs = torch.sigmoid(probs_logits)  # --> [0, 1]
    probs = torch.clip(probs, EPS, 1 - EPS)

    # temperature scheduling: temp = temp * {args.tempdecay} every {args.tempfreq} epochs
    if args.temp_freq > 0 and epoch > 0 and epoch % args.temp_freq == 0:
        softmax_temp = softmax_temp * args.temp_decay
    # regratio scheduling: regratio = regratio * {args.regratiodecay} every {args.regratiofreq} epochs
    if args.regratio_freq > 0 and epoch > 0 and epoch % args.regratio_freq == 0:
        regratio = regratio * args.regratio_decay
    
    egn_beta = reg_val * regratio

    probs_copy = probs.clone().detach()
    
    # alignment
    
    # 1. soft derandomization
    probs_new = torch.clip(probs, EPS, 1 - EPS)
    if to_align:
        if args.seq_align:
            # sequential derandomization
            for i_node in range(n_points):
                vx2results = incremental_differences(
                    probs_new,
                    dist,
                    k,
                    egn_beta,
                    s_max=s_max,
                    small_card_th=small_card_th,
                    small_card_ratio=small_card_ratio,
                )
                vx2results_i = vx2results[i_node]
                probs_new = alignment_greedy_flatten_single_node(
                    probs_new, i_node, vx2results_i, softmax_func, sign_func
                )
                probs_new = torch.clip(probs_new, EPS, 1 - EPS)
        else:
            # greedy derandomization
            for _ in range(args.align_iter):
                vx2results = incremental_differences(
                    probs_new,
                    dist,
                    k,
                    egn_beta,
                    s_max=s_max,
                    small_card_th=small_card_th,
                    small_card_ratio=small_card_ratio,
                )
                probs_new = alignment_greedy_flatten(
                    probs_new, vx2results, softmax_func, sign_func
                )
                probs_new = torch.clip(probs_new, EPS, 1 - EPS)

    # 2. soft k-means
    if to_soft_kmeans:
        probs_new = soft_kmeans_func(probs_new)
        probs_new = torch.clip(probs_new, EPS, 1 - EPS)
    
    # 3. soft top-k
    if to_soft_topk:
        probs_new = soft_topk_func(probs_new)
        probs_new = torch.clip(probs_new, EPS, 1 - EPS)
    
    probs_new_copy = probs_new.clone().detach()
    
    obj = compute_objective_differentiable_exact(dist, probs_new)
    card_dist = pmf_poibin_vec(probs_new, device, use_normalization=False)
    k_diff = torch.abs(
        torch.arange(probs_new.shape[0] + 1, device=device) - args.n_choose
    )
    # avoid empty output
    k_diff[:small_card_th] *= small_card_ratio
    constraint_conflict = (card_dist * k_diff).sum()
    obj += reg_val * constraint_conflict
    obj.mean().backward()
    optimizer.step()
    optimizer.zero_grad()
    
    train_obj_this_round = obj.item()
    test_obj_this_round = []

    # test
    with torch.inference_mode():
        probs_output = probs_copy.clone().detach()
        (
            test_obj,
            selected_indices,
            finish_time,
            best_objective_list,        
        ) = egn_pb_sequential_facility_location_direct_probs_no_kmeans_clean(
            points,
            graph,
            dist,
            args.n_choose,
            probs_output,
            egn_beta=reg_val,
            time_limit=-1,
            distance_metric="euclidean",
            s_max=s_max,
            small_card_th=small_card_th,
            small_card_ratio=small_card_ratio,
            test_each_round=True,
        )
        test_obj_this_round.append(test_obj)
        
        probs_output = probs_copy.clone().detach()
        (
            test_obj,
            selected_indices,
            finish_time,
            best_objective_list,        
        ) = egn_pb_greedy_facility_location_direct_probs_no_kmeans_clean(
            points,
            graph,
            dist,
            args.n_choose,
            probs_output,
            egn_beta=reg_val,
            time_limit=-1,
            distance_metric="euclidean",
            s_max=s_max,
            small_card_th=small_card_th,
            small_card_ratio=small_card_ratio,
            test_each_round=True,
        )
        test_obj_this_round.append(test_obj)
        
        
        if False:
            probs_output = probs_copy.clone().detach()
            (
                test_obj,
                selected_indices,
                finish_time,
                best_objective_list,        
            ) = egn_pb_sequential_facility_location_direct_probs_no_kmeans_clean(
                points,
                graph,
                dist,
                args.n_choose,
                probs_output,
                egn_beta=reg_val,
                time_limit=-1,
                distance_metric="euclidean",
                s_max=s_max,
                small_card_th=small_card_th,
                small_card_ratio=small_card_ratio,
                test_each_round=False,
            )
            test_obj_this_round.append(test_obj)
            
            probs_output = probs_copy.clone().detach()
            (
                test_obj,
                selected_indices,
                finish_time,
                best_objective_list,
            ) = egn_pb_sequential_facility_location_direct_probs_kmeans_clean(        
                points,
                graph,
                dist,
                args.n_choose,
                probs_output,
                egn_beta=reg_val,
                time_limit=-1,
                distance_metric="euclidean",
                s_max=s_max,
                small_card_th=small_card_th,
                small_card_ratio=small_card_ratio,
                test_each_round=True,
            )
            test_obj_this_round.append(test_obj)
            
            probs_output = probs_copy.clone().detach()
            (
                test_obj,
                selected_indices,
                finish_time,
                best_objective_list,
            ) = egn_pb_sequential_facility_location_direct_probs_kmeans_clean(        
                points,
                graph,
                dist,
                args.n_choose,
                probs_output,
                egn_beta=reg_val,
                time_limit=-1,
                distance_metric="euclidean",
                s_max=s_max,
                small_card_th=small_card_th,
                small_card_ratio=small_card_ratio,
                test_each_round=False,
            )
            test_obj_this_round.append(test_obj)
            
            # greedy            
            
            probs_output = probs_copy.clone().detach()
            (
                test_obj,
                selected_indices,
                finish_time,
                best_objective_list,        
            ) = egn_pb_greedy_facility_location_direct_probs_no_kmeans_clean(
                points,
                graph,
                dist,
                args.n_choose,
                probs_output,
                egn_beta=reg_val,
                time_limit=-1,
                distance_metric="euclidean",
                s_max=s_max,
                small_card_th=small_card_th,
                small_card_ratio=small_card_ratio,
                test_each_round=False,
            )
            test_obj_this_round.append(test_obj)
            
            probs_output = probs_copy.clone().detach()
            (
                test_obj,
                selected_indices,
                finish_time,
                best_objective_list,
            ) = egn_pb_greedy_facility_location_direct_probs_kmeans_clean(        
                points,
                graph,
                dist,
                args.n_choose,
                probs_output,
                egn_beta=reg_val,
                time_limit=-1,
                distance_metric="euclidean",
                s_max=s_max,
                small_card_th=small_card_th,
                small_card_ratio=small_card_ratio,
                test_each_round=True,
            )
            test_obj_this_round.append(test_obj)
            
            probs_output = probs_copy.clone().detach()
            (
                test_obj,
                selected_indices,
                finish_time,
                best_objective_list,
            ) = egn_pb_greedy_facility_location_direct_probs_kmeans_clean(        
                points,
                graph,
                dist,
                args.n_choose,
                probs_output,
                egn_beta=reg_val,
                time_limit=-1,
                distance_metric="euclidean",
                s_max=s_max,
                small_card_th=small_card_th,
                small_card_ratio=small_card_ratio,
                test_each_round=False,
            )
            test_obj_this_round.append(test_obj)

    print(epoch, train_obj_this_round, *test_obj_this_round)
