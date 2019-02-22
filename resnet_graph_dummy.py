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
    'ibuf':            num_cols * iprec * 2048 / 2,
    'obuf':            num_rows * oprec * 2048 / 2,
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

with graph.as_default():
	with graph.name_scope('inputs'):
		i = get_tensor(shape=(batch_size,224,224,3), name='data', trainable=False)

	with graph.name_scope('conv1'):
		weights = get_tensor(shape=(64, 7, 7, 3),
		             name='weights')
		biases = get_tensor(shape=(64),
		             name='biases')
		conv1 = conv2D(i, weights, biases, stride=(1,2,2,1), pad='SAME')

		gamma = get_tensor(shape=(1,112,112,64), name='gamma')
		beta = get_tensor(shape=(1,112,112,64), name='beta')
		conv1 = b_norm(conv1,gamma,beta)

		conv1 = reLU(conv1)
		conv1_pool = maxPool(conv1, pooling_kernel=(1,3,3,1), stride=(1,2,2,1), pad='VALID')
		conv1_pool = fork(conv1_pool)

#################### conv2a #######################	
	with graph.name_scope('conv2a_1'):
		weights = get_tensor(shape=(256, 1, 1, 64),
		             name='weights')
		biases = get_tensor(shape=(256),
		             name='biases')
		conv2a_1 = conv2D(conv1_pool, weights, biases, stride=(1,1,1,1), pad='SAME')  

		gamma = get_tensor(shape=(1,55,55,256), name='gamma')
		beta = get_tensor(shape=(1,55,55,256), name='beta')
		conv2a_1 = b_norm(conv2a_1,gamma,beta)

	with graph.name_scope('conv2a_2a'):
		weights = get_tensor(shape=(64, 1, 1, 64),
		             name='weights')
		biases = get_tensor(shape=(64),
		             name='biases')
		conv2a_2a = conv2D(conv1_pool, weights, biases, stride=(1,1,1,1), pad='SAME')    

		gamma = get_tensor(shape=(1,55,55,64), name='gamma')
		beta = get_tensor(shape=(1,55,55,64), name='beta')
		conv2a_2a = b_norm(conv2a_2a,gamma,beta)
		conv2a_2a = reLU(conv2a_2a)

	with graph.name_scope('conv2a_2b'):
		weights = get_tensor(shape=(64, 3, 3, 64),
		             name='weights')
		biases = get_tensor(shape=(64),
		             name='biases')
		conv2a_2b = conv2D(conv2a_2a, weights, biases, stride=(1,1,1,1), pad='SAME')
		gamma = get_tensor(shape=(1,55,55,64), name='gamma')
		beta = get_tensor(shape=(1,55,55,64), name='beta')
		conv2a_2b = b_norm(conv2a_2b,gamma,beta)
		conv2a_2b = reLU(conv2a_2b)

	with graph.name_scope('conv2a_2c'):
		weights = get_tensor(shape=(256, 1, 1, 64),
		             name='weights')
		biases = get_tensor(shape=(256),
		             name='biases')
		conv2a_2c = conv2D(conv2a_2b, weights, biases, stride=(1,1,1,1), pad='SAME')
		gamma = get_tensor(shape=(1,55,55,256), name='gamma')
		beta = get_tensor(shape=(1,55,55,256), name='beta')
		conv2a_2c = b_norm(conv2a_2c,gamma,beta)
		
	with graph.name_scope('res2a'):
		res2a = add([conv2a_1, conv2a_2c])
		res2a = reLU(res2a)
		res2a_pool = maxPool(res2a, pooling_kernel=(1,55,55,1), stride=(1,1,1,1), pad='VALID')

	with graph.name_scope('fc'):
		weights = get_tensor(shape=(1000, 1, 1, 256),
				     name='weights')
		biases = get_tensor(shape=(1000),
				     name='biases')
		fc = fc(res2a_pool, weights, biases, stride=(1,1,1,1), pad='SAME')

compile_graph('resnet50/', graph, acc_obj)
#compile_graph_bp('resnet50_bp/', graph, acc_obj)

f = open("resnet50/resnet50.bin", "rb")
decode_instr(f)






