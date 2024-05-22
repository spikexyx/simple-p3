# simple-p3
A simple implementation of the paper P3: Distributed Deep Graph Learning at Scale

## Prerequisites
CUDA, Pytorch, Python: environment used in this implementation is CUDA 12.1.1, Pytorch 2.2.1, Python 3.11
Install DGL: refer to dgl site https://www.dgl.ai/pages/start.html
Required libraries:
~~~
pip install pydantic ogb torchmetrics matplotlib seaborn  # and other libs may be required
~~~

## Usage
Available args:
~~~
python base_run.py -h # or python p3_run.py -h
~~~

Run baseline:
~~~
python base_run.py [--args]
~~~

Run P3:
~~~
python p3_run.py [--args]
~~~

For example, running p3 on 4 gpus with model sage and with batch_size 256, using hidden feature size 64 on graph ogbn-products:
~~~
python p3_run.py --model sage --batch_size 256 --hid_feats 64 --nprocs 4 --graph_name ogbn-products
~~~
