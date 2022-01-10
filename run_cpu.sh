batch=12
log_path_opt=""
lr=1e-4

while getopts d:l:b:r: flag
do
    case "${flag}" in
        d) data=${OPTARG};;
        l) log_path_opt="-log_path "${OPTARG};;
        b) batch=${OPTARG};;
        r) lr=${OPTARG};;
    esac
done


n_head=4
n_layers=4
d_model=512
d_rnn=64
d_inner=1024
d_k=512
d_v=512
dropout=0.1
smooth=0.1
epoch=100

python Main.py -data $data -batch $batch -n_head $n_head -n_layers $n_layers -d_model $d_model -d_rnn $d_rnn -d_inner $d_inner -d_k $d_k -d_v $d_v -dropout $dropout -lr $lr -smooth $smooth -epoch $epoch -device cpu $log_path_opt
