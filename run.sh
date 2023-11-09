#!/bin/bash
python3 -m torch.distributed.launch --nproc-per-node=8 ISSM_DGL_distributed.py --epochs=200 --model-type='mlp' --batch-size=16 --base-lr=0.01 --out-ch=3 --mesh=5000
python3 -m torch.distributed.launch --nproc-per-node=8 ISSM_DGL_distributed.py --epochs=200 --model-type='egcn' --batch-size=16 --base-lr=0.01 --out-ch=3 --mesh=5000
python3 -m torch.distributed.launch --nproc-per-node=8 ISSM_DGL_distributed.py --epochs=200 --model-type='gcn' --batch-size=16 --base-lr=0.01 --out-ch=3 --mesh=5000
python3 -m torch.distributed.launch --nproc-per-node=8 ISSM_DGL_distributed.py --epochs=200 --model-type='gat' --batch-size=16 --base-lr=0.01 --out-ch=3 --mesh=5000

python3 -m torch.distributed.launch --nproc-per-node=8 ISSM_DGL_distributed.py --epochs=200 --model-type='mlp' --batch-size=16 --base-lr=0.01 --out-ch=3 --mesh=10000
python3 -m torch.distributed.launch --nproc-per-node=8 ISSM_DGL_distributed.py --epochs=200 --model-type='egcn' --batch-size=16 --base-lr=0.01 --out-ch=3 --mesh=10000
python3 -m torch.distributed.launch --nproc-per-node=8 ISSM_DGL_distributed.py --epochs=200 --model-type='gcn' --batch-size=16 --base-lr=0.01 --out-ch=3 --mesh=10000
python3 -m torch.distributed.launch --nproc-per-node=8 ISSM_DGL_distributed.py --epochs=200 --model-type='gat' --batch-size=16 --base-lr=0.01 --out-ch=3 --mesh=10000

python3 -m torch.distributed.launch --nproc-per-node=8 ISSM_DGL_distributed.py --epochs=200 --model-type='mlp' --batch-size=16 --base-lr=0.01 --out-ch=3 --mesh=20000
python3 -m torch.distributed.launch --nproc-per-node=8 ISSM_DGL_distributed.py --epochs=200 --model-type='egcn' --batch-size=16 --base-lr=0.01 --out-ch=3 --mesh=20000
python3 -m torch.distributed.launch --nproc-per-node=8 ISSM_DGL_distributed.py --epochs=200 --model-type='gcn' --batch-size=16 --base-lr=0.01 --out-ch=3 --mesh=20000
python3 -m torch.distributed.launch --nproc-per-node=8 ISSM_DGL_distributed.py --epochs=200 --model-type='gat' --batch-size=16 --base-lr=0.01 --out-ch=3 --mesh=20000
