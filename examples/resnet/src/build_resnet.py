import os
import csv

import numpy as np

import mxnet as mx
from mxnet.gluon.model_zoo.vision import get_model
from mxnet.gluon.utils import download

import tvm
from tvm.contrib import graph_runtime, cc
import nnvm

batch_size = 1
opt_level = 3
target = tvm.target.create("llvm")
image_shape = (3, 224, 224)
data_shape = (batch_size,) + image_shape

def build(target_dir):

	block = get_model('resnet18_v1', pretrained=True)
	
	sym, params = nnvm.frontend.from_mxnet(block)
	net = nnvm.sym.softmax(sym)

	with nnvm.compiler.build_config(opt_level=opt_level):
		graph, lib, params = nnvm.compiler.build(
			net, target, shape={"data": data_shape}, params=params)

	lib.save(os.path.join(target_dir, "deploy_lib.o"))
	cc.create_shared(os.path.join(target_dir, "deploy_lib.so"),
    				[os.path.join(target_dir, "deploy_lib.o")])
	
	with open(os.path.join(target_dir, "deploy_graph.json"), "w") as fo:
	    fo.write(graph.json())
	with open(os.path.join(target_dir,"deploy_param.params"), "wb") as fo:
	    fo.write(nnvm.compiler.save_param_dict(params))

	img_name = 'cat.png'
	synset_url = ''.join(['https://gist.githubusercontent.com/zhreshold/',
                      '4d0b62f3d01426887599d4f7ede23ee5/raw/',
                      '596b27d23537e5a1b5751d2b0481ef172f58b539/',
                      'imagenet1000_clsid_to_human.txt'])
	synset_name = 'synset.txt'
	download('https://github.com/dmlc/mxnet.js/blob/master/data/cat.png?raw=true', img_name)
	download(synset_url, synset_name)

	with open(synset_name) as fin:
		synset = eval(fin.read())

	with open("synset.csv", "w") as fout:
		w = csv.writer(fout)
		w.writerows(synset.items())

def test_build(target_dir):
	graph = open(os.path.join(target_dir, "deploy_graph.json")).read()
	lib = tvm.module.load(os.path.join(target_dir, "deploy_lib.so"))
	params = bytearray(open(os.path.join(target_dir,"deploy_param.params"), "rb").read())
	input_data = tvm.nd.array(np.random.uniform(size=data_shape).astype("float32"))
	ctx = tvm.cpu()
	module = graph_runtime.create(graph, lib, ctx)
	module.load_params(params)
	module.run(data=input_data)

	out = module.get_output(0).asnumpy()

if __name__ == '__main__':
	import sys
	import logging
	logger = logging.getLogger(__name__)

	if len(sys.argv) != 2:
		sys.exit(-1)

	logger.info("building the model")
	build(sys.argv[1])
	logger.info("build was successful!")
	
	logger.info("testing the build")
	test_build(sys.argv[1])
	logger.info("test was successful")