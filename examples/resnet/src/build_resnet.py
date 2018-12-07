import os

import numpy as np

import tvm
from tvm.contrib import graph_runtime, cc
import nnvm.compiler
import nnvm.testing

batch_size = 1
opt_level = 3
target = tvm.target.create("llvm")
image_shape = (3, 224, 224)
data_shape = (batch_size,) + image_shape

def build(target_dir):
	net, params = nnvm.testing.resnet.get_workload(
		num_layers=18, batch_size=batch_size, image_shape=image_shape)

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

def test_build(target_dir):
	ctx = tvm.cpu()

	loaded_json = open(os.path.join(target_dir, "deploy_graph.json")).read()
	loaded_lib = tvm.module.load(os.path.join(target_dir, "deploy_lib.so"))
	loaded_params = bytearray(open(os.path.join(target_dir,"deploy_param.params"), "rb").read())
	
	input_data = tvm.nd.array(np.random.uniform(size=data_shape).astype("float32"))

	module = graph_runtime.create(loaded_json, loaded_lib, ctx)
	module.load_params(loaded_params)
	module.run(data=input_data)
	
	out = module.get_output(0).asnumpy()

if __name__ == '__main__':
	import sys
	if len(sys.argv) != 2:
		sys.exit(-1)
	print("building ...")
	build(sys.argv[1])
	print("build was successful!")
	print("testing the build ...")
	test_build(sys.argv[1])
	print("test was successful")