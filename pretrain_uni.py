import numpy
import os

from nmt_uni import train

def main(job_id, params):
    print params
    validerr = train(saveto=params['model'][0],
                     reload_=params['reload'][0],
                     dim_word=params['dim_word'][0],
                     dim=params['dim'][0],
                     n_words=params['n-words'][0],
                     n_words_src=params['n-words'][0],
                     decay_c=params['decay-c'][0],
                     clip_c=params['clip-c'][0],
                     lrate=params['learning-rate'][0],
                     optimizer=params['optimizer'][0], 
                     patience=1000,
                     maxlen=50,
                     batch_size=64,
                     valid_batch_size=64,
                     validFreq=1000,
                     dispFreq=50,
                     saveFreq=1000,
                     sampleFreq=99,
                     datasets=['/misc/kcgscratch1/ChoGroup/junyoung_exp/wmt15/ruen/train/all_ru-en.en.tok.bpe',
                               '/misc/kcgscratch1/ChoGroup/junyoung_exp/wmt15/ruen/train/all_ru-en.ru.tok.bpe'],
                     valid_datasets=['/misc/kcgscratch1/ChoGroup/junyoung_exp/wmt15/ruen/dev/newstest2013-src.en.tok.bpe',
                                     '/misc/kcgscratch1/ChoGroup/junyoung_exp/wmt15/ruen/dev/newstest2013-ref.ru.tok.bpe'],
                     dictionaries=['/misc/kcgscratch1/ChoGroup/junyoung_exp/wmt15/ruen/train/all_ru-en.en.tok.bpe.word.pkl',
                               '/misc/kcgscratch1/ChoGroup/junyoung_exp/wmt15/ruen/train/all_ru-en.ru.tok.bpe.word.pkl'],
                     use_dropout=params['use-dropout'][0])
    return validerr

if __name__ == '__main__':
    main(0, {
        'model': ['models/model_wmt15_bpe2k_uni_en-ru.npz'],
        'dim_word': [512],
        'dim': [1028],
        'n-words': [20000], 
        'optimizer': ['adadelta'],
        'decay-c': [0.], 
        'clip-c': [1.], 
        'use-dropout': [False],
        'learning-rate': [0.0001],
        'reload': [False]})


