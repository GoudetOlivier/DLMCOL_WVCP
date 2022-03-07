#!/bin/bash

rm -rf venv Evol solutions logs

mkdir Evol solutions logs

python3 -m venv venv
source venv/bin/activate
python3 -m pip install --upgrade pip
python3 -m pip install -r requirements.txt
