lr=1e-1
reg=1e-1
sct=0
smax=10
gpu=$1

program=fl_single_graph.py
setting=fl_orig_100
mkdir -p logs/${setting} -m 777
CUDA_VISIBLE_DEVICES=$gpu python $program --n_nodes 100 --n_choose 10 --no_align --lr $lr --reg $reg --no_topk --no_kmeans --smax $smax --sct $sct >>logs/${setting}/lr${lr}_reg${reg}_orig.txt
