---
title: conda、jupyterlab使用技巧
# description: 
date: 2018-09-12
# slug: 
# image: 
categories:
    - Python
---
### conda
创建环境:
```bash
conda create -n py370 python=3.70
```
删除环境
```bash
conda env remove --name py370
```

### jupyter-lab
jupyter-lab默认使用base环境，如果想在创建notebook时使用自己其他的环境，执行以下命令即可。执行完要重启jupyter-lab
```bash
# linux
source activate py370

# win
conda activate py370

pip install jupyter notebook ipykernel

ipython kernel install --user --name=py70
```