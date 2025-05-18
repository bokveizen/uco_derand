lr=1e-1
reg=1e-1

temp_list=(
    10.0
    1.0
    1e-1
    1e-2
    1e-3
)

sct=0
smax=10
gpu=$1


program=fl_single_graph.py

for temp in "${temp_list[@]}"; do
    setting=fl_grd_100
    mkdir -p logs/${setting} -m 777
    log_path=logs/${setting}/lr${lr}_reg${reg}_temp${temp}.txt
    # skip the existing files
    if [ -f $log_path ]; then
        continue
    fi
    CUDA_VISIBLE_DEVICES=$gpu python $program --n_nodes 100 --n_choose 10 --align_iter 200 --lr $lr --reg $reg --temp $temp --no_topk --no_kmeans --smax $smax --sct $sct >> $log_path
done