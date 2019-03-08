import logging
import numpy as np
import array

from dnnweaver2.benchmarks import get_graph
from dnnweaver2.simulator.accelerator import Accelerator
from dnnweaver2.compiler import *
from dnnweaver2.fpga.fpgamanager import FPGAManager

from dnnweaver2.scalar.dtypes import FixedPoint

from compiler import compile_lstm, compile_lstm_bp
from compiler_helpers import decode_instr

graph = Graph('LSTM', dataset='imagenet', log_level=logging.INFO)

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


H = 1024
X = 1

with graph.as_default():
	with graph.name_scope('inputs'):
		z = get_tensor(shape=(batch_size,1,1,H+X), name='z', trainable=False)
		Cin = get_tensor(shape=(batch_size,1,1,H), name='Cin', trainable=False)

	with graph.name_scope('cell_state'):
		Cin = cell_state(Cin)
		z = cell_state(z)

	with graph.name_scope('inner_fc'):
		wf = get_tensor(shape=(H, 1, 1, H+X),
				     name='wf')
		bf = get_tensor(shape=(H),
				     name='bf')

		wi = get_tensor(shape=(H, 1, 1, H+X),
				     name='wi')
		bi = get_tensor(shape=(H),
				     name='bi')

		wc = get_tensor(shape=(H, 1, 1, H+X),
				     name='wc')
		bc = get_tensor(shape=(H),
				     name='bc')

		wo = get_tensor(shape=(H, 1, 1, H+X),
				     name='wo')
		bo = get_tensor(shape=(H),
				     name='bo')

		of,oi,oc,oo = lstm_fc(z, [wf,wi,wc,wo], [bf,bi,bc,bo], stride=(1,1,1,1), pad='SAME')

	with graph.name_scope('arithmetics'):
		Cout, hout = lstm_arith(Cin, of, oi, oc, oo)
		hout1, hout2 = fork(hout)

	with graph.name_scope('output'):
		weights = get_tensor(shape=(X, 1, 1, H),
				     name='weights')
		biases = get_tensor(shape=(X),
				     name='biases')
		fc = fc(hout1, weights, biases, stride=(1,1,1,1), pad='SAME')

compile_lstm('lstm/', graph, acc_obj)
compile_lstm_bp('lstm_bp/', graph, acc_obj)








