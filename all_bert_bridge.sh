#!/usr/bin/env bash

#-----------------------------------------------restaurant-----------------------------------------------
python train_bert_bridge.py  --source service --target restaurant --use_unlabel 1 --use_prototype 1 --name BERT+CNN+BRIDGE
python train_bert_bridge.py  --source laptop --target restaurant --use_unlabel 1 --use_prototype 1 --name BERT+CNN+BRIDGE
python train_bert_bridge.py  --source device --target restaurant --use_unlabel 1 --use_prototype 1 --name BERT+CNN+BRIDGE

#-----------------------------------------------service-----------------------------------------------
python train_bert_bridge.py  --source restaurant --target service --use_unlabel 1 --use_prototype 1 --name BERT+CNN+BRIDGE
python train_bert_bridge.py  --source laptop --target service --use_unlabel 1 --use_prototype 1 --name BERT+CNN+BRIDGE
python train_bert_bridge.py  --source device --target service --use_unlabel 1 --use_prototype 1 --name BERT+CNN+BRIDGE

#-----------------------------------------------laptop-----------------------------------------------
python train_bert_bridge.py  --source restaurant --target laptop --use_unlabel 1 --use_prototype 1 --name BERT+CNN+BRIDGE
python train_bert_bridge.py  --source service --target laptop --use_unlabel 1 --use_prototype 1 --name BERT+CNN+BRIDGE

#-----------------------------------------------device-----------------------------------------------
python train_bert_bridge.py  --source restaurant --target device --use_unlabel 1 --use_prototype 1 --name BERT+CNN+BRIDGE
python train_bert_bridge.py  --source service --target device --use_unlabel 1 --use_prototype 1 --name BERT+CNN+BRIDGE






