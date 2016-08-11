THEANO_FLAGS=device=gpu$2 python simultrans_training.py --sample 32 --batchsize 1 --target $1 --gamma $3 --recurrent True  2>&1 | tee .log/$4.log
