#! /bin/bash

rm -f ./cookies.txt
touch ./cookies.txt
wget --load-cookies ./cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies ./cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=0B5MzpY9kBtDVZ2RpVDYwWmxoSUk' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=0B5MzpY9kBtDVZ2RpVDYwWmxoSUk" -O 20170512-110547.zip
rm -rf ./cookies.txt
unzip 20170512-110547.zip
rm -f 20170512-110547.zip
rm -f ./cookies.txt
cp -r 20170512-110547 20170512-110547_mvds
cd 20170512-110547_mvds
mv model-20170512-110547.ckpt-250000.data-00000-of-00001 facenet_movidius.data-00000-of-00001
mv model-20170512-110547.ckpt-250000.index facenet_movidius.index
mv model-20170512-110547.meta facenet_movidius.meta
python3 ../convert_facenet.py model_base=facenet_movidius
cd facenet_movidius_ncs
mvNCCompile facenet_movidius_ncs.meta -w facenet_movidius_ncs -s 12 -in input -on output -o facenet_movidius_ncs.graph
cd ..
mv facenet_movidius_ncs/facenet_movidius_ncs.graph facenet_movidius_ncs.graph
rm -rf facenet_movidius_ncs
rm 20170512-110547.pb
rm facenet_movidius.data-00000-of-00001
rm facenet_movidius.index
rm facenet_movidius.meta
