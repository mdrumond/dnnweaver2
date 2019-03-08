import logging
import numpy as np
import array

from dnnweaver2.benchmarks import get_graph
from dnnweaver2.simulator.accelerator import Accelerator
from dnnweaver2.compiler import *
from dnnweaver2.fpga.fpgamanager import FPGAManager

from dnnweaver2.scalar.dtypes import FixedPoint

from compiler import compile_graph, compile_graph_bp
from compiler_helpers import decode_instr


from keras.applications.resnet50 import ResNet50
import keras




def get_inbound_tensor(l,output_tensors,out_ind=0):
	no_inbound = len(l._inbound_nodes[0].inbound_layers)
	if no_inbound==1:
		out_tensor = output_tensors[l._inbound_nodes[0].inbound_layers[0].name]
		if isinstance(out_tensor, list):
			out_ind += 1
			return out_tensor[out_ind-1]
		else:
			return out_tensor
	elif no_inbound>1:
		layers = []
		for k in l._inbound_nodes[0].inbound_layers:
			out_tensor = output_tensors[k.name]
			if isinstance(out_tensor, list):
				out_ind += 1
				layers.append(out_tensor[out_ind-1])
			else:
				layers.append(out_tensor)
		return layers
	else:
		assert False, 'No inbound layer'

resnet_model = ResNet50()

graph = Graph('Resnet50', dataset='imagenet', log_level=logging.INFO)

acc_prec = 16
iprec = acc_prec
wprec = acc_prec
bprec = 32
oprec = 64
prec = (iprec, wprec, bprec, oprec)

DRAM_BW = 256 #bits per cycle

num_rows = 2
num_cols = 2

bram = {
    'ibuf':            10*num_cols * iprec * 2048 / 2,
    'obuf':            10*num_rows * oprec * 2048 / 2,
    'wbuf': num_cols * num_rows * wprec *  512 / 2,
    'bbuf':            num_rows * bprec * 2048 / 2,
}


acc_obj = Accelerator(
    N=num_rows, M=num_cols,
    prec=prec,
    mem_if_width=DRAM_BW,
    frequency=150e6,
    sram=bram
)

batch_size = 4


output_tensors={}
out_ind = 0
with graph.as_default():
	for l in resnet_model.layers:
		print('Layer_name:{}'.format(l.name))
		
		if isinstance(l,keras.engine.topology.InputLayer):
			with graph.name_scope(l.name):
				out_tensor = get_tensor(shape=(batch_size,l.input.shape[1].value,l.input.shape[2].value,l.input.shape[3].value), name=l.name+'data', trainable=False)
				output_tensors[l.name] = out_tensor

		elif isinstance(l,keras.layers.convolutional.ZeroPadding2D):
			output_tensors[l.name] = get_inbound_tensor(l,output_tensors)

		elif isinstance(l,keras.layers.convolutional.Conv2D):
			with graph.name_scope(l.name):
				input_tensor = get_inbound_tensor(l,output_tensors,out_ind)
				weights = get_tensor(shape=(l.kernel.shape[3].value, l.kernel.shape[0].value, l.kernel.shape[1].value, l.kernel.shape[2].value),
					     name=l.name+'weights')
				biases = get_tensor(shape=(l.kernel.shape[3].value),
					     name=l.name+'biases')
				out_tensor = conv2D(input_tensor, weights, biases, stride=(1,l.strides[0],l.strides[1],1), pad='SAME')
				output_tensors[l.name] = out_tensor

		elif isinstance(l,keras.layers.normalization.BatchNormalization):
			with graph.name_scope(l.name):
				input_tensor = get_inbound_tensor(l,output_tensors)
				gamma = get_tensor(shape=(l.gamma.shape[0].value), name=l.name+'gamma')
				beta = get_tensor(shape=(l.beta.shape[0].value), name=l.name+'beta')
				out_tensor = b_norm(input_tensor,gamma,beta)
				output_tensors[l.name] = out_tensor

		elif isinstance(l,keras.layers.core.Activation):
			with graph.name_scope(l.name):
				input_tensor = get_inbound_tensor(l,output_tensors)
				out_tensor = reLU(input_tensor)
				output_tensors[l.name] = out_tensor

		elif isinstance(l,keras.layers.pooling.MaxPooling2D):
			with graph.name_scope(l.name):
				input_tensor = get_inbound_tensor(l,output_tensors)
				out_tensor = maxPool(input_tensor, pooling_kernel=(1,l.pool_size[0],l.pool_size[1],1), stride=(1,l.strides[0],l.strides[1],1), pad='VALID')
				output_tensors[l.name] = out_tensor

		#for some bug in keras, isinstance doesnt work for Add
		elif 'keras.layers.merge.Add' in l.__str__():
			with graph.name_scope(l.name):
				input_tensor1, input_tensor2 = get_inbound_tensor(l,output_tensors,out_ind)
				out_tensor = add([input_tensor1, input_tensor2])
				output_tensors[l.name] = out_tensor

		elif isinstance(l,keras.layers.pooling.AveragePooling2D):
			with graph.name_scope(l.name):
				input_tensor = get_inbound_tensor(l,output_tensors)
				out_tensor = averagePool(input_tensor, pooling_kernel=(1,l.pool_size[0],l.pool_size[1],1), stride=(1,l.strides[0],l.strides[1],1), pad='VALID')
				output_tensors[l.name] = out_tensor

		#TODO: Discuss with Mario if skipping this layer would cause any memory banking issues
		elif isinstance(l,keras.layers.core.Flatten):
			output_tensors[l.name] = get_inbound_tensor(l,output_tensors)

		elif isinstance(l,keras.layers.core.Dense):
			with graph.name_scope(l.name):
				input_tensor = get_inbound_tensor(l,output_tensors)
				weights = get_tensor(shape=(l.kernel.shape[1].value, 1, 1, l.kernel.shape[0].value),
					     name=l.name+'weights')
				biases = get_tensor(shape=(l.kernel.shape[1].value),
					     name=l.name+'biases')
				out_tensor = fc(input_tensor, weights, biases, stride=(1,1,1,1), pad='SAME')
				output_tensors[l.name] = out_tensor

		else:
			assert False, 'Layer not implemented!'

		if len(l._outbound_nodes)>1:
			with graph.name_scope(l.name):
				out_tensor1, out_tensor2 = fork(output_tensors[l.name])
				output_tensors[l.name] = [out_tensor1,out_tensor2]
				out_ind = 0

	graph.print_ops()
	compile_graph('resnet50/', graph, acc_obj)
	compile_graph_bp('resnet50_bp/', graph, acc_obj)

#f = open("resnet50/resnet50.bin", "rb")
#decode_instr(f)






