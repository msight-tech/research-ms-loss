#!/bin/bash
set -e

CUB_ROOT='resource/datasets/CUB_200_2011/'
CUB_DATA='http://www.vision.caltech.edu.s3-us-west-2.amazonaws.com/visipedia-data/CUB-200-2011/CUB_200_2011.tgz'


if [[ ! -d "${CUB_ROOT}" ]]; then
    mkdir -p resource/datasets
    pushd resource/datasets
    echo "Downloading CUB_200_2011 data-set..."
    wget ${CUB_DATA}
    tar -zxf CUB_200_2011.tgz
    popd
fi
# Generate train.txt and test.txt splits
echo "Generating the train.txt/test.txt split files"
python scripts/split_cub_for_ms_loss.py


