THEANO_FLAGS=device=gpu$2 python simultrans_evaluation.py  --sinit 1 --target 0.5 --sample 64 --batchsize 1 --Rtype $1 --gamma 1 --id $3 --recurrent True  2>&1 | tee .images/$4.log
