#!/bin/bash

filename="fastcc_tensors.tar.gz"

if [[ ! -f "$filename" ]]; then
  wget -O fastcc_tensors.tar.gz https://zenodo.org/records/15891225/files/fastcc_tensors.tar.gz?download=1
  tar -xvzf fastcc_tensors.tar.gz
else
  echo "$filename already exists, skipping curl."
fi


