#!/bin/bash

if [ "$#" -ne 2 ]; then
	echo "Usage: "
	echo "./train_sv_classifier.sh 'embeddings_file' 'pickle_file'"
	exit
fi

embeddings_file=$1
sv_pickle_file=$2
python3 ../big_fiubrother_classifier/classifier_support_vector.py TRAIN $embeddings_file $sv_pickle_file
