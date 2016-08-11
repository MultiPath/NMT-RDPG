#!/bin/bash

ref=" /misc/kcgscratch1/ChoGroup/junyoung_exp/wmt15/ruen/dev/newstest2013-ref.ru.tok"
sed -i 's/@@ //g' $1
./data/multi-bleu.perl ref < $1
