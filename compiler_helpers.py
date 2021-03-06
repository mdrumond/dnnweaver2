import math
import logging
import numpy as np

from dnnweaver2.compiler import GraphCompiler
from dnnweaver2.benchmarks import get_graph
from dnnweaver2.graph import Graph
from dnnweaver2 import get_tensor
from dnnweaver2.scalar.dtypes import FQDtype, FixedPoint
from dnnweaver2.tensorOps.cnn import conv2D,conv2D_bp, maxPool, Convolution, flatten, matmul
from dnnweaver2.utils.utils import ceil_a_by_b
from dnnweaver2.optimizer.optimizer import OP_TYPE

from custom_isa import OPCodes, ACTType, MACType

tile_deps = {}
tile_deps['B/b']   = {'ibuf': True,  'wbuf': False, 'obuf': True,  'bbuf': False}
tile_deps['OW/ow'] = {'ibuf': True,  'wbuf': False, 'obuf': True,  'bbuf': False}
tile_deps['OH/oh'] = {'ibuf': True,  'wbuf': False, 'obuf': True,  'bbuf': False}
tile_deps['IC/ic'] = {'ibuf': True,  'wbuf': True,  'obuf': False, 'bbuf': False}
tile_deps['OC/oc'] = {'ibuf': False, 'wbuf': True,  'obuf': True,  'bbuf': True}
tile_deps['G/g'] = {'ibuf': False, 'wbuf': True,  'obuf': True,  'bbuf': True}

def pe_ind(pe_x, pe_y, num_rows):
	return pe_y*num_rows + pe_x

def encode_instr(opcode, op1=None, op2=None, op3=None):
	opstr = '{0:08b}'.format(opcode)
	assert len(opstr) <= 8, 'opcode overflow: {}'.format(opcode)
	instr = [int(opstr,2)]

	if op1 is not None:
		op1str = '{0:032b}'.format(op1)
		assert len(op1str) <= 32, 'op1 overflow: {}'.format(op1)
		instr += [int(op1str[0:8],2), int(op1str[8:16],2), int(op1str[16:24],2), int(op1str[24:],2)]

	if op2 is not None:
		op2str = '{0:032b}'.format(op2)
		assert len(op2str) <= 32, 'op2 overflow: {}'.format(op2)		
		instr += [int(op2str[0:8],2), int(op2str[8:16],2), int(op2str[16:24],2), int(op2str[24:],2)]

	if op3 is not None:
		op3str = '{0:08b}'.format(op3)
		assert len(op3str) <= 8, 'op3 overflow: {}'.format(op3)
		instr += [int(op3str,2)]

	return instr

def read_op(f,no_bytes):
	bin_str = ''
	for i in range(no_bytes):
		bin_str += '{0:08b}'.format(ord(f.read(1)))
	return int(bin_str,2)

def decode_instr(f):
	rbin = f.read(1)
	while len(rbin)>0:
		opcode = ord(rbin)

		if opcode==OPCodes.DRAM_RD:
			op1 = read_op(f,4)
			op2 = read_op(f,4)
			print('DRAM_RD {} {}'.format(op1, op2))
		elif opcode==OPCodes.DRAM_WR:
			op1 = read_op(f,4)
			op2 = read_op(f,4)
			print('DRAM_WR {} {}'.format(op1, op2))
		elif opcode==OPCodes.DRAM_RD_TP:
			op1 = read_op(f,4)
			op2 = read_op(f,4)
			op3 = read_op(f,1)
			print('DRAM_RD_TP {} {} {}'.format(op1, op2, op3))
		elif opcode==OPCodes.MAC:
			op1 = read_op(f,4)
			op2 = read_op(f,4)
			print('MAC {} {}'.format(op1, op2))
		elif opcode==OPCodes.MULT:
			op1 = read_op(f,4)
			print('MULT {}'.format(op1))
		elif opcode==OPCodes.MULT_DER:
			op1 = read_op(f,4)
			print('MULT_DER {}'.format(op1))
		elif opcode==OPCodes.ADD:
			op1 = read_op(f,4)
			print('ADD {}'.format(op1))
		elif opcode==OPCodes.ACT:
			op1 = read_op(f,4)
			op2 = read_op(f,4)
			print('ACT {} {}'.format(op1, op2))
		elif opcode==OPCodes.POOL:
			op1 = read_op(f,4)
			op2 = read_op(f,4)
			print('POOL {} {}'.format(op1, op2))
		elif opcode==OPCodes.POOL_BP:
			op1 = read_op(f,4)
			op2 = read_op(f,4)
			print('POOL_BP {} {}'.format(op1, op2))
		elif opcode==OPCodes.BNORM:
			op1 = read_op(f,4)
			op2 = read_op(f,4)
			print('BNORM {} {}'.format(op1, op2))
		elif opcode==OPCodes.BNORM_BP:
			op1 = read_op(f,4)
			op2 = read_op(f,4)
			print('BNORM_BP {} {}'.format(op1, op2))
		elif opcode==OPCodes.NOP:
			op1 = read_op(f,4)
			print('NOP {}'.format(op1))
		elif opcode==OPCodes.EOL:
			print('EOL')
		else:
			assert False, 'OPCODE not defined. Something is wrong!'

		rbin = f.read(1)

def find_tile_id(ic_ind, oc_ind, oh_ind, ow_ind, b_ind, g_ind, num_IC, num_OC, num_OH, num_OW, num_B, num_G, block):
	if block=='ibuf':
		return oh_ind + num_OH*ow_ind + num_OH*num_OW*ic_ind + num_OH*num_OW*num_IC*b_ind

	elif block=='wbuf':
		return num_OH*num_OW*num_IC*num_B + ic_ind + num_IC*oc_ind + num_IC*num_OC*g_ind

	elif block=='bbuf':
		return num_OH*num_OW*num_IC*num_B + num_IC*num_OC*num_G + oc_ind + num_OC*g_ind

	elif block=='obuf':
		return num_OH*num_OW*num_IC*num_B + num_IC*num_OC*num_G + num_OC*num_G + oh_ind + num_OH*ow_ind + num_OH*num_OW*oc_ind + num_OH*num_OW*num_OC*b_ind + num_OH*num_OW*num_OC*num_B*g_ind

def conv2instr(instr, stats, dir_name, layer_name, tensor_ids, conv_params, op_type=None, mac_type=MACType.CONV, activation=None):
	acc_obj, K, O, S, IC, OC, B, G, _ = conv_params
	OH = O
	OW = O
	IH = OH*S
	IW = OW*S
	KH = K
	KW = K

	num_rows = acc_obj.N
	num_cols = acc_obj.M
	DRAM_BW = acc_obj.mem_if_width
	iprec, wprec, bprec, oprec = acc_obj.prec

	tiling = stats.tiling
	order = stats.order

	print('O:{} I:{} K:{} IC:{} OC:{} B:{} G:{}'.format(O,IH,K,IC,OC,B,G))
	print('Order and tiling:')
	for o in order:
		print('{}: {}-{}'.format(o,tiling[o][0],tiling[o][1]))

	num_IC,ic = tiling['IC/ic']
	num_OC,oc = tiling['OC/oc']
	num_OH,oh = tiling['OH/oh']
	num_OW,ow = tiling['OW/ow']
	num_B,b = tiling['B/b']
	num_G,g = tiling['G/g']

	ih = (oh-1)*S+K
	iw = (ow-1)*S+K

	if op_type==OP_TYPE.FW or op_type==OP_TYPE.LSTM_FW:
		block_transpose = {'obuf':(False,OH), 'ibuf':(False,IH), 'wbuf':(False,K), 'bbuf':(False,None)}
		output_block = 'obuf'
	elif op_type==OP_TYPE.GD or op_type==OP_TYPE.LSTM_GD:
		block_transpose = {'obuf':(True,OH), 'ibuf':(False,IH), 'wbuf':(False,K), 'bbuf':(False,None)}
		output_block = 'wbuf'	
	elif op_type==OP_TYPE.BP or op_type==OP_TYPE.LSTM_BP:	
		block_transpose = {'obuf':(False,OH), 'ibuf':(False,IH), 'wbuf':(True,K), 'bbuf':(False,None)}	
		output_block = 'ibuf'
	else:
		assert False

	memory_fetch = {'obuf':True, 'ibuf':True, 'wbuf':True, 'bbuf':True}
	memory_fetch[output_block] = False

	if op_type==OP_TYPE.FW or op_type==OP_TYPE.GD or op_type==OP_TYPE.BP:
		data_id, weight_id, bias_id, out_id = tensor_ids
		tensor_id = {'ibuf':data_id, 'wbuf':weight_id, 'bbuf':bias_id, 'obuf':out_id}
	elif op_type==OP_TYPE.LSTM_FW or op_type==OP_TYPE.LSTM_GD or op_type==OP_TYPE.LSTM_BP:
		data_id, wf_id, wi_id, wc_id, wo_id, bf_id, bi_id, bc_id, bo_id, of_id, oi_id, oc_id, oo_id = tensor_ids
		if bf_id is not None or bi_id is not None or bc_id is not None or bo_id is not None:
			tensor_id = {'ibuf':[data_id], 'wbuf':[wf_id, wi_id, wc_id, wo_id], 'bbuf':[bf_id, bi_id, bc_id, bo_id], 'obuf':[of_id, oi_id, oc_id, oo_id]}		
		else:
			tensor_id = {'ibuf':[data_id], 'wbuf':[wf_id, wi_id, wc_id, wo_id], 'bbuf':None, 'obuf':[of_id, oi_id, oc_id, oo_id]}
	else:
		assert False

	tiles = np.zeros( (num_OH*num_OW*num_IC*num_B + num_IC*num_OC*num_G + num_OC*num_G + num_OH*num_OW*num_OC*num_B*num_G) )

	mac_cnt = 0
	dram_read = 0
	dram_write = 0

	ftext= open(dir_name+layer_name+".txt","w")

	for n in range(tiling[order[5]][0]):
		if order[5]=='OC/oc':
			oc_ind = n
		if order[5]=='B/b':
			b_ind = n
		if order[5]=='OW/ow':
			ow_ind = n
		if order[5]=='OH/oh':
			oh_ind = n
		if order[5]=='IC/ic':
			ic_ind = n
		if order[5]=='G/g':
			g_ind = n

		for i in range(tiling[order[4]][0]):
			if order[4]=='OC/oc':
				oc_ind = i
			if order[4]=='B/b':
				b_ind = i
			if order[4]=='OW/ow':
				ow_ind = i
			if order[4]=='OH/oh':
				oh_ind = i
			if order[4]=='IC/ic':
				ic_ind = i
			if order[4]=='G/g':
				g_ind = i

			for j in range(tiling[order[3]][0]):
				if order[3]=='OC/oc':
					oc_ind = j
				if order[3]=='B/b':
					b_ind = j
				if order[3]=='OW/ow':
					ow_ind = j
				if order[3]=='OH/oh':
					oh_ind = j
				if order[3]=='IC/ic':
					ic_ind = j
				if order[3]=='G/g':
					g_ind = j

				for k in range(tiling[order[2]][0]):
					if order[2]=='OC/oc':
						oc_ind = k
					if order[2]=='B/b':
						b_ind = k
					if order[2]=='OW/ow':
						ow_ind = k
					if order[2]=='OH/oh':
						oh_ind = k
					if order[2]=='IC/ic':
						ic_ind = k
					if order[2]=='G/g':
						g_ind = k

					for l in range(tiling[order[1]][0]):
						if order[1]=='OC/oc':
							oc_ind = l
						if order[1]=='B/b':
							b_ind = l
						if order[1]=='OW/ow':
							ow_ind = l
						if order[1]=='OH/oh':
							oh_ind = l
						if order[1]=='IC/ic':
							ic_ind = l
						if order[1]=='G/g':
							g_ind = l

						for m in range(tiling[order[0]][0]):
							if order[0]=='OC/oc':
								oc_ind = m
							if order[0]=='B/b':
								b_ind = m
							if order[0]=='OW/ow':
								ow_ind = m
							if order[0]=='OH/oh':
								oh_ind = m
							if order[0]=='IC/ic':
								ic_ind = m
							if order[0]=='G/g':
								g_ind = m

							_oc = min(oc,OC-oc_ind*oc)
							_b = min(b,B-b_ind*b)
							_ow = min(ow,OW-ow_ind*ow)
							_oh = min(oh,OH-oh_ind*oh)
							_ic = min(ic,IC-ic_ind*ic)

							kw = K
							kh = K
							_iw = K + (_ow - 1) * S
							_ih = K + (_oh - 1) * S

							data_tile_size = _iw*_ih*_ic*_b*iprec
							weight_tile_size = kw*kh*_ic*_oc*g*wprec
							bias_tile_size = _oc*g*bprec
							output_tile_size = _oh*_ow*_oc*_b*g*oprec
							
							tile_size = {'obuf':output_tile_size, 'ibuf':data_tile_size, 'wbuf':weight_tile_size, 'bbuf':bias_tile_size}

							if memory_fetch[output_block]==True:
								dram_write += tile_size[output_block]
								instr += conv_mem_instr(ftext, g_ind, g, tensor_id[output_block], op_type, 'DRAM_WR', block=output_block, size=tile_size[output_block], K=None)

							for f in memory_fetch:
								if memory_fetch[f]==True and tensor_id[f] is not None:
									if f==output_block:
										out_tile_id = find_tile_id(ic_ind, oc_ind, oh_ind, ow_ind, b_ind, g_ind, num_IC, num_OC, num_OH, num_OW, num_B, num_G, f)
										if tiles[out_tile_id]>0: #Skip reading output tile if it is first time
											dram_read += tile_size[f]
											if block_transpose[f][0]==True:
												instr += conv_mem_instr(ftext, g_ind, g, tensor_id[f], op_type, 'DRAM_RD_TP', block=f, size=tile_size[f], K=block_transpose[f][1])
											else:
												instr += conv_mem_instr(ftext, g_ind, g, tensor_id[f], op_type, 'DRAM_RD', block=f, size=tile_size[f], K=None)
     										tiles[out_tile_id] += 1
									else:
										dram_read += tile_size[f]
										if block_transpose[f][0]==True:
											instr += conv_mem_instr(ftext, g_ind, g, tensor_id[f], op_type, 'DRAM_RD_TP', block=f, size=tile_size[f], K=block_transpose[f][1])
										else:
											instr += conv_mem_instr(ftext, g_ind, g, tensor_id[f], op_type, 'DRAM_RD', block=f, size=tile_size[f], K=None)
									memory_fetch[f] = False
									

							_compute_cycles_per_pe, _instr = compute_instr(ftext, _ih, _iw, _ic, kh, kw, _oh, _ow, _oc, _b, g, g_ind, ic_ind, num_IC, num_rows, num_cols, mac_type, activation, op_type=op_type)				
							mac_cnt += num_rows * num_cols * _compute_cycles_per_pe
							instr += _instr
							


							for f in memory_fetch:
								if tile_deps[order[0]][f] == True and tiling[order[0]][0]>1:
									memory_fetch[f] = True

						for f in memory_fetch:
							if tile_deps[order[1]][f] == True and tiling[order[1]][0]>1:
								memory_fetch[f] = True

					for f in memory_fetch:
						if tile_deps[order[2]][f] == True and tiling[order[2]][0]>1:
							memory_fetch[f] = True

				for f in memory_fetch:
					if tile_deps[order[3]][f] == True and tiling[order[3]][0]>1:
						memory_fetch[f] = True

			for f in memory_fetch:
				if tile_deps[order[4]][f] == True and tiling[order[4]][0]>1:
					memory_fetch[f] = True

		for f in memory_fetch:
			if tile_deps[order[5]][f] == True and tiling[order[5]][0]>1:
				memory_fetch[f] = True

	dram_write += tile_size[output_block]
	instr += conv_mem_instr(ftext, g_ind, g, tensor_id[output_block], op_type, 'DRAM_WR', block=output_block, size=tile_size[output_block], K=None)

	ftext.write('EOL #End of layer\n')
	instr += encode_instr(OPCodes.EOL)
	ftext.close()

	if op_type==OP_TYPE.FW or op_type==OP_TYPE.LSTM_FW:
		exp_mac_cnt = KH * KW * OH * OW * ceil_a_by_b(IC, acc_obj.N) * acc_obj.N *  ceil_a_by_b(OC, acc_obj.M) * acc_obj.M * B * G
	elif op_type==OP_TYPE.GD or op_type==OP_TYPE.LSTM_GD:
		exp_mac_cnt = OH * OW * KH * KW * ceil_a_by_b(IC, acc_obj.N) * acc_obj.N *  ceil_a_by_b(OC, acc_obj.M) * acc_obj.M * B * G
	elif op_type==OP_TYPE.BP or op_type==OP_TYPE.LSTM_BP:
		exp_mac_cnt = (ih*num_OH) * (iw*num_OW) * KH * KW * ceil_a_by_b(IC, acc_obj.N) * acc_obj.N *  ceil_a_by_b(OC, acc_obj.M) * acc_obj.M * B * G

	
	#if S==1: #With stride this estimation is not simple
	assert mac_cnt == exp_mac_cnt, 'MAC count:{:,} Expected:{:,}. Something is wrong!'.format(mac_cnt, exp_mac_cnt)
	
	return instr


def compute_instr(ftext, _ih, _iw, _ic, kh, kw, _oh, _ow, _oc, _b, g, g_ind, ic_ind, num_IC, num_rows, num_cols, mac_type, activation, op_type=None):
	instr = []

	if op_type==OP_TYPE.FW or op_type==OP_TYPE.LSTM_FW:
		compute_cycles_per_pe = math.ceil(float(_ic)/num_rows) * math.ceil(float(_oc)/num_cols) * kw * kh * _oh * _ow * _b
	elif op_type==OP_TYPE.GD or op_type==OP_TYPE.LSTM_GD:	
		compute_cycles_per_pe = math.ceil(float(_ic)/num_rows) * math.ceil(float(_oc)/num_cols) * kw * kh * _oh * _ow * _b		
	elif op_type==OP_TYPE.BP or op_type==OP_TYPE.LSTM_BP:
		compute_cycles_per_pe = math.ceil(float(_ic)/num_rows) * math.ceil(float(_oc)/num_cols) * kw * kh * _ih * _iw * _b
	else:
		assert False

	mac_cnt=0
	if op_type==OP_TYPE.FW or op_type==OP_TYPE.GD or op_type==OP_TYPE.BP:
		ftext.write('ALU {} {} #Compute inner loop \n'.format(int(compute_cycles_per_pe), mac_type))
		instr += encode_instr(OPCodes.MAC, int(compute_cycles_per_pe), mac_type)

		if activation is not None and ic_ind == num_IC-1:
			ftext.write('ACT {} {} #Apply activation function \n'.format(activation, math.ceil(float(_oc)/num_cols) * _oh * _ow * _b))
			instr += encode_instr(OPCodes.ACT, math.ceil(float(_oc)/num_cols) * _oh * _ow * _b, activation)

		mac_cnt += compute_cycles_per_pe
	else:
		for _g in range(g):
			ftext.write('ALU {} {} #Compute inner loop \n'.format(int(compute_cycles_per_pe), mac_type))
			instr += encode_instr(OPCodes.MAC, int(compute_cycles_per_pe), mac_type)
			mac_cnt += compute_cycles_per_pe

			if activation is not None and ic_ind == num_IC-1:
				ftext.write('ACT {} {} #Apply activation function \n'.format(activation[_g+g_ind*g], int(math.ceil(float(_oc)/num_cols))))

				instr += encode_instr(OPCodes.ACT, int(math.ceil(float(_oc)/num_cols) * _oh * _ow * _b), activation[_g+g_ind*g])
			
			

	return mac_cnt, instr



def conv_mem_instr(ftext, g_ind, g, tensor_id, op_type, mem_op, block, size=0, K=None):
	instr = []

	if op_type==OP_TYPE.FW or op_type==OP_TYPE.BP or op_type==OP_TYPE.GD:
		ftext.write('{} {} {} #Memory op for {} tile\n'.format(mem_op, size, tensor_id, block))
		if mem_op is 'DRAM_WR':
			instr += encode_instr(OPCodes.DRAM_WR, size, tensor_id)
		elif mem_op is 'DRAM_RD':
			instr += encode_instr(OPCodes.DRAM_RD, size, tensor_id)
		elif mem_op is 'DRAM_RD_TP':
			assert K is not None, 'K value is not entered.'
			instr += encode_instr(OPCodes.DRAM_RD_TP, size, tensor_id, K)

	elif op_type==OP_TYPE.LSTM_FW or op_type==OP_TYPE.LSTM_BP or op_type==OP_TYPE.LSTM_GD:
		if block=='ibuf':
			ids = tensor_id
			ftext.write('{} {} {} #Memory op for {} tile\n'.format(mem_op, size, ids[0], block))
			if mem_op is 'DRAM_WR':
				instr += encode_instr(OPCodes.DRAM_WR, size, ids[0])
			elif mem_op is 'DRAM_RD':
				instr += encode_instr(OPCodes.DRAM_RD, size, ids[0])
			elif mem_op is 'DRAM_RD_TP':
				assert K is not None, 'K value is not entered.'
				instr += encode_instr(OPCodes.DRAM_RD_TP, size, ids[0], K)

		else:
			ids = tensor_id[g_ind*g:(g_ind+1)*g]

			for i in range(g):
				ftext.write('{} {} {} #Memory op for {} tile\n'.format(mem_op, size, ids[i], block))
				if mem_op is 'DRAM_WR':
					instr += encode_instr(OPCodes.DRAM_WR, size, ids[i])
				elif mem_op is 'DRAM_RD':
					instr += encode_instr(OPCodes.DRAM_RD, size, ids[i])
				elif mem_op is 'DRAM_RD_TP':
					assert K is not None, 'K value is not entered.'
					instr += encode_instr(OPCodes.DRAM_RD_TP, size, ids[i], K)

	return instr

def lstm_mem_instr1111(f, g_ind, g, ids, mem_op='DRAM_WR', size=0, K=None):
	instr = []

	ids = ids[g_ind*g:(g_ind+1)*g]

	for i in range(g):
		f.write('{} {} {} #Memory op\n'.format(mem_op, size, ids[i]))
		if mem_op is 'DRAM_WR':
			instr += encode_instr(OPCodes.DRAM_WR, size, ids[i])
		elif mem_op is 'DRAM_RD':
			instr += encode_instr(OPCodes.DRAM_RD, size, ids[i])
		elif mem_op is 'DRAM_RD_TP':
			assert K is not None, 'K value is not entered.'
			instr += encode_instr(OPCodes.DRAM_RD_TP, size, ids[i], K)

	return instr


def lstm2instr(instr, stats, dir_name, layer_name, tensor_ids, OH, OW, OC, IC, B, K, S, acc_obj, mac_type=MACType.CONV, activation=None, transpose_weight=False):
	IH = OH*S
	IW = OW*S

	data_id, wf_id, wi_id, wc_id,  wo_id, bf_id, bi_id, bc_id, bo_id, of_id, oi_id, oc_id, oo_id = tensor_ids

	num_rows = acc_obj.N
	num_cols = acc_obj.M
	DRAM_BW = acc_obj.mem_if_width
	iprec, wprec, bprec, oprec = acc_obj.prec

	tiling = stats.tiling
	order = stats.order

	print('Order and tiling:')
	for o in order:
		print('{}: {}-{}'.format(o,tiling[o][0],tiling[o][1]))

	ic = tiling['IC/ic'][1]
	oc = tiling['OC/oc'][1]
	oh = tiling['OH/oh'][1]
	ow = tiling['OW/ow'][1]
	b = tiling['B/b'][1]
	g = tiling['G/g'][1]

	obuf_fetch = False
	ibuf_fetch = True
	wbuf_fetch = True
	bbuf_fetch = True

	retire_output = False

	cycle = 0
	dram_read = 0
	dram_write = 0


	out_tiles_access = np.zeros((tiling['OH/oh'][0],tiling['OW/ow'][0],tiling['OC/oc'][0],tiling['B/b'][0],tiling['G/g'][0]))

	f= open(dir_name+layer_name+".txt","w")

	for n in range(tiling[order[5]][0]):
		if order[5]=='OC/oc':
			oc_ind = n
		if order[5]=='B/b':
			b_ind = n
		if order[5]=='OW/ow':
			ow_ind = n
		if order[5]=='OH/oh':
			oh_ind = n
		if order[5]=='IC/ic':
			ic_ind = n
		if order[5]=='G/g':
			g_ind = n

		for i in range(tiling[order[4]][0]):
			if order[4]=='OC/oc':
				oc_ind = i
			if order[4]=='B/b':
				b_ind = i
			if order[4]=='OW/ow':
				ow_ind = i
			if order[4]=='OH/oh':
				oh_ind = i
			if order[4]=='IC/ic':
				ic_ind = i
			if order[4]=='G/g':
				g_ind = i

			for j in range(tiling[order[3]][0]):
				if order[3]=='OC/oc':
					oc_ind = j
				if order[3]=='B/b':
					b_ind = j
				if order[3]=='OW/ow':
					ow_ind = j
				if order[3]=='OH/oh':
					oh_ind = j
				if order[3]=='IC/ic':
					ic_ind = j
				if order[3]=='G/g':
					g_ind = j

				for k in range(tiling[order[2]][0]):
					if order[2]=='OC/oc':
						oc_ind = k
					if order[2]=='B/b':
						b_ind = k
					if order[2]=='OW/ow':
						ow_ind = k
					if order[2]=='OH/oh':
						oh_ind = k
					if order[2]=='IC/ic':
						ic_ind = k
					if order[2]=='G/g':
						g_ind = k

					for l in range(tiling[order[1]][0]):
						if order[1]=='OC/oc':
							oc_ind = l
						if order[1]=='B/b':
							b_ind = l
						if order[1]=='OW/ow':
							ow_ind = l
						if order[1]=='OH/oh':
							oh_ind = l
						if order[1]=='IC/ic':
							ic_ind = l
						if order[1]=='G/g':
							g_ind = l

						for m in range(tiling[order[0]][0]):
							if order[0]=='OC/oc':
								oc_ind = m
							if order[0]=='B/b':
								b_ind = m
							if order[0]=='OW/ow':
								ow_ind = m
							if order[0]=='OH/oh':
								oh_ind = m
							if order[0]=='IC/ic':
								ic_ind = m
							if order[0]=='G/g':
								g_ind = m							

							if ic_ind == tiling['IC/ic'][0]-1:
								retire_output = True

							_oc = min(oc,OC-oc_ind*oc)
							_b = min(b,B-b_ind*b)
							_ow = min(ow,OW-ow_ind*ow)
							_oh = min(oh,OH-oh_ind*oh)
							_ic = min(ic,IC-ic_ind*ic)

							kw = K
							kh = K
							_iw = K + (_ow - 1) * S
							_ih = K + (_oh - 1) * S
							data_tile_size = _iw*_ih*_ic*_b*iprec
							weight_tile_size = kw*kh*_ic*_oc*wprec
							bias_tile_size = _oc*bprec
							output_tile_size = _oh*_ow*_oc*_b*oprec

							if obuf_fetch == True:
								dram_write += output_tile_size*g
								instr += lstm_mem_instr(f, g_ind, g, [of_id,oi_id,oc_id,oo_id], 'DRAM_WR', output_tile_size)
								cycle += int(math.ceil(float(output_tile_size*g) / DRAM_BW))

								if out_tiles_access[oh_ind,ow_ind,oc_ind,b_ind,g_ind]>0: #Skip reading output tile if it is first time
									dram_read += output_tile_size*g
									instr += lstm_mem_instr(f, g_ind, g, [of_id,oi_id,oc_id,oo_id], 'DRAM_RD', output_tile_size)
									cycle += int(math.ceil(float(output_tile_size*g) / DRAM_BW))
								
								out_tiles_access[oh_ind,ow_ind,oc_ind,b_ind,g_ind] = out_tiles_access[oh_ind,ow_ind,oc_ind,b_ind,g_ind] + 1

								obuf_fetch = False

							if ibuf_fetch == True:
								dram_read += data_tile_size

								f.write('DRAM_RD {} {} #Read input tile\n'.format(data_tile_size, data_id))
								instr += encode_instr(OPCodes.DRAM_RD, data_tile_size, data_id)

								cycle += int(math.ceil(float(data_tile_size) / DRAM_BW))
								ibuf_fetch = False

							if wbuf_fetch == True:
								dram_read += weight_tile_size
								if transpose_weight == True:
									instr += lstm_mem_instr(f, g_ind, g, [wf_id,wi_id,wc_id,wo_id], 'DRAM_RD_TP', weight_tile_size, K)
									f.write('DRAM_RD_TP {} {} {} #Read weight tile\n'.format(weight_tile_size, weight_id, K))
									instr += encode_instr(OPCodes.DRAM_RD_TP, weight_tile_size, weight_id, K)
								else:
									instr += lstm_mem_instr(f, g_ind, g, [wf_id,wi_id,wc_id,wo_id], 'DRAM_RD', weight_tile_size)
								cycle += int(math.ceil(float(weight_tile_size) / DRAM_BW))
								wbuf_fetch = False

							if bbuf_fetch == True:
								dram_read += bias_tile_size
								instr += lstm_mem_instr(f, g_ind, g, [bf_id,bi_id,bc_id,bo_id], 'DRAM_RD', bias_tile_size)
								cycle += int(math.ceil(float(bias_tile_size) / DRAM_BW))
								bbuf_fetch = False
							
							
							compute_cycles_per_pe = math.ceil(float(_ic)/num_rows) * math.ceil(float(_oc)/num_cols) * kw * kh * _oh * _ow * _b
							
							for _g in range(g):
								f.write('ALU {} #Compute inner loop \n'.format(int(compute_cycles_per_pe)))
								instr += encode_instr(OPCodes.MAC, int(compute_cycles_per_pe), mac_type)

								if retire_output==True:
									f.write('ACT {} {} #Apply activation function \n'.format(int(math.ceil(float(_oc)/num_cols)), lstm_activations[_g+g_ind*g]))
									instr += encode_instr(OPCodes.ACT, int(math.ceil(float(_oc)/num_cols)), lstm_activations[_g+g_ind*g])

							cycle += math.ceil(float(_ic)/num_rows) * math.ceil(float(_oc)/num_cols) * kw * kh * _oh * _ow * _b  * g
							if retire_output==True:
								cycle += 2 * math.ceil(float(_oc)/num_cols) * _oh * _ow * _b * g
								retire_output = False

							if tile_deps_lstm[order[0]]['obuf'] == True:
								obuf_fetch = True
							if tile_deps_lstm[order[0]]['ibuf'] == True:
								ibuf_fetch = True
							if tile_deps_lstm[order[0]]['wbuf'] == True:
								wbuf_fetch = True
							if tile_deps_lstm[order[0]]['bbuf'] == True:
								bbuf_fetch = True

						if tile_deps_lstm[order[1]]['obuf'] == True:
							obuf_fetch = True
						if tile_deps_lstm[order[1]]['ibuf'] == True:
							ibuf_fetch = True
						if tile_deps_lstm[order[1]]['wbuf'] == True:
							wbuf_fetch = True
						if tile_deps_lstm[order[1]]['bbuf'] == True:
							bbuf_fetch = True

					if tile_deps_lstm[order[2]]['obuf'] == True:
						obuf_fetch = True
					if tile_deps_lstm[order[2]]['ibuf'] == True:
						ibuf_fetch = True
					if tile_deps_lstm[order[2]]['wbuf'] == True:
						wbuf_fetch = True
					if tile_deps_lstm[order[2]]['bbuf'] == True:
						bbuf_fetch = True

				if tile_deps_lstm[order[3]]['obuf'] == True:
					obuf_fetch = True
				if tile_deps_lstm[order[3]]['ibuf'] == True:
					ibuf_fetch = True
				if tile_deps_lstm[order[3]]['wbuf'] == True:
					wbuf_fetch = True
				if tile_deps_lstm[order[3]]['bbuf'] == True:
					bbuf_fetch = True

			if tile_deps_lstm[order[4]]['obuf'] == True:
				obuf_fetch = True
			if tile_deps_lstm[order[4]]['ibuf'] == True:
				ibuf_fetch = True
			if tile_deps_lstm[order[4]]['wbuf'] == True:
				wbuf_fetch = True
			if tile_deps_lstm[order[4]]['bbuf'] == True:
				bbuf_fetch = True

		if tile_deps_lstm[order[5]]['obuf'] == True:
			obuf_fetch = True
		if tile_deps_lstm[order[5]]['ibuf'] == True:
			ibuf_fetch = True
		if tile_deps_lstm[order[5]]['wbuf'] == True:
			wbuf_fetch = True
		if tile_deps_lstm[order[5]]['bbuf'] == True:
			bbuf_fetch = True

	dram_write += output_tile_size*g
	instr += lstm_mem_instr(f, g_ind, g, [of_id,oi_id,oc_id,oo_id], 'DRAM_WR', output_tile_size)
	cycle += int(math.ceil(float(output_tile_size*g) / DRAM_BW))

	f.write('EOL #End of layer\n')
	instr += encode_instr(OPCodes.EOL)
	f.close()

	print('Total DRAM read simulated: {:,}'.format(dram_read))
	print('Total DRAM write simulated: {:,}'.format(dram_write))
	print('Total cycles simulated: {:,}'.format(cycle))

	print('Memory cycles: {:,}'.format(stats.memory_cycles_required))
	print('Compute cycles: {:,}'.format(stats.compute_cycles))

	return instr

def lstm_arithm_instr(instr, dir_name, name, tensor_ids, H, acc_obj):
	Cin_id, of_id, oi_id, oc_id, oo_id, Cout_id, hout_id = tensor_ids

	N = acc_obj.N
	M = acc_obj.M
	oprec = acc_obj.prec[3]

	input_output_size = H*oprec
	
	r=1 #large r means larger data chunks, less instructions.
	mem_chunk = N*M*oprec*r
	
	f= open(dir_name+name+".txt","w")

	mem_cnt = 0
	while mem_cnt < input_output_size:
		f.write('DRAM_RD {} {} #Read vec1\n'.format(mem_chunk, Cin_id))
		instr += encode_instr(OPCodes.DRAM_RD, mem_chunk, Cin_id)
		f.write('DRAM_RD {} {} #Read vec2\n'.format(mem_chunk, of_id))
		instr += encode_instr(OPCodes.DRAM_RD, mem_chunk, of_id)
		f.write('ALU {} {} #Vector op \n'.format('Mult', r))
		instr += encode_instr(OPCodes.MULT, r)

		f.write('DRAM_RD {} {} #Read vec1\n'.format(mem_chunk, oi_id))
		instr += encode_instr(OPCodes.DRAM_RD, mem_chunk, oi_id)
		f.write('DRAM_RD {} {} #Read vec2\n'.format(mem_chunk, oc_id))
		instr += encode_instr(OPCodes.DRAM_RD, mem_chunk, oc_id)
		f.write('ALU {} {} #Vector op \n'.format('Mult', r))
		instr += encode_instr(OPCodes.MULT, r)

		f.write('ALU {} {} #Vector op \n'.format('Add', r))
		instr += encode_instr(OPCodes.ADD, r)

		f.write('DRAM_WR {} {} #Write back output\n'.format(mem_chunk, Cout_id))
		instr += encode_instr(OPCodes.DRAM_WR, mem_chunk, Cout_id)
		
		f.write('ACT {} #Apply activation function \n'.format('TANH'))
		instr += encode_instr(OPCodes.ACT, r, ACTType.TANH)			

		f.write('DRAM_RD {} {} #Read vec1\n'.format(mem_chunk, oo_id))
		instr += encode_instr(OPCodes.DRAM_RD, mem_chunk, oo_id)

		f.write('ALU {} {} #Vector op \n'.format('Mult', r))
		instr += encode_instr(OPCodes.MULT, r)

		f.write('DRAM_WR {} {} #Write back output\n'.format(mem_chunk, hout_id))
		instr += encode_instr(OPCodes.DRAM_WR, mem_chunk, hout_id)

		mem_cnt += mem_chunk

	f.write('EOL #End of layer\n')
	instr += encode_instr(OPCodes.EOL)

	return instr

def lstm_arithm_bp_instr(instr, dir_name, name, tensor_ids, H, acc_obj):
	Cin_id, Cin_bp_id, of_id, of_bp_id, oi_id, oi_bp_id, oc_id, oc_bp_id, oo_id, oo_bp_id, Cout_id, Cout_bp_id, hout_id, hout_bp_id = tensor_ids

	N = acc_obj.N
	M = acc_obj.M
	oprec = acc_obj.prec[3]

	input_output_size = H*oprec
	
	r=1 #large r means larger data chunks, less instructions.
	mem_chunk = N*M*oprec*r
	
	f= open(dir_name+name+".txt","w")

	mem_cnt = 0
	while mem_cnt < input_output_size:
		f.write('DRAM_RD {} {} #Read hout_bp\n'.format(mem_chunk, hout_bp_id))
		instr += encode_instr(OPCodes.DRAM_RD, mem_chunk, hout_bp_id)

		f.write('DRAM_RD {} {} #Read Cout\n'.format(mem_chunk, Cout_id))
		instr += encode_instr(OPCodes.DRAM_RD, mem_chunk, Cout_id)

		f.write('DRAM_RD {} {} #Read Cout_bp\n'.format(mem_chunk, Cout_bp_id))
		instr += encode_instr(OPCodes.DRAM_RD, mem_chunk, Cout_bp_id)

		f.write('DRAM_RD {} {} #Read oo\n'.format(mem_chunk, oo_id))
		instr += encode_instr(OPCodes.DRAM_RD, mem_chunk, oo_id)

		f.write('DRAM_RD {} {} #Read oc\n'.format(mem_chunk, oc_id))
		instr += encode_instr(OPCodes.DRAM_RD, mem_chunk, oc_id)

		f.write('DRAM_RD {} {} #Read oi\n'.format(mem_chunk, oi_id))
		instr += encode_instr(OPCodes.DRAM_RD, mem_chunk, oi_id)

		f.write('DRAM_RD {} {} #Read of\n'.format(mem_chunk, of_id))
		instr += encode_instr(OPCodes.DRAM_RD, mem_chunk, of_id)

		f.write('DRAM_RD {} {} #Read Cin\n'.format(mem_chunk, Cin_id))
		instr += encode_instr(OPCodes.DRAM_RD, mem_chunk, Cin_id)


		f.write('ACT {} #TANH(Cout) \n'.format('TANH'))
		instr += encode_instr(OPCodes.ACT, r, ACTType.TANH)

		f.write('ALU {} {} #hout_bp*TANH(Cout)->Oo_bp \n'.format('Mult', r))
		instr += encode_instr(OPCodes.MULT, r)


		f.write('ACT {} #SIGMOID_INV(Oo) -> Zo \n'.format('SIGMOID_INV'))
		instr += encode_instr(OPCodes.ACT, r, ACTType.SIGMOID_INV)

		f.write('ACT {} #SIGMOID_DER(Zo) \n'.format('SIGMOID_DER'))
		instr += encode_instr(OPCodes.ACT, r, ACTType.SIGMOID_DER)

		f.write('ALU {} {} #Oo_bp*SIGMOID_DER(Zo)->Zo_bp \n'.format('Mult', r))
		instr += encode_instr(OPCodes.MULT, r)

		f.write('DRAM_WR {} {} #Write Oo_bp\n'.format(mem_chunk, oo_bp_id))
		instr += encode_instr(OPCodes.DRAM_WR, mem_chunk, oo_bp_id)



		f.write('ALU {} {} #hout_bp*Oo \n'.format('Mult', r))
		instr += encode_instr(OPCodes.MULT, r)

		f.write('ACT {} #TANH_DER(Cout) \n'.format('TANH_DER'))
		instr += encode_instr(OPCodes.ACT, r, ACTType.TANH_DER)

		f.write('ALU {} {} #hout_bp*Oo*TANH_DER(Cout) \n'.format('Mult', r))
		instr += encode_instr(OPCodes.MULT, r)

		f.write('ALU {} {} #hout_bp*Oo*TANH_DER(Cout)+Cout_bp -> tmp_res \n'.format('Add', r))
		instr += encode_instr(OPCodes.ADD, r)



		f.write('ALU {} {} #tmp_res*oi -> oc_bp \n'.format('Mult', r))
		instr += encode_instr(OPCodes.MULT, r)		

		f.write('ACT {} #TANH_INV(Oc) -> Zc \n'.format('TANH_INV'))
		instr += encode_instr(OPCodes.ACT, r, ACTType.TANH_INV)

		f.write('ACT {} #TANH_DER(Zc) \n'.format('TANH_DER'))
		instr += encode_instr(OPCodes.ACT, r, ACTType.SIGMOID_DER)

		f.write('ALU {} {} #Oo_bp*SIGMOID_DER(Zo)->Zo_bp \n'.format('Mult', r))
		instr += encode_instr(OPCodes.MULT, r)

		f.write('DRAM_WR {} {} #Write back oc_bp\n'.format(mem_chunk, oc_bp_id))
		instr += encode_instr(OPCodes.DRAM_WR, mem_chunk, oc_bp_id)


		f.write('ALU {} {} #tmp_res*oc \n'.format('Mult', r))
		instr += encode_instr(OPCodes.MULT, r)	

		f.write('ACT {} #SIGMOID_INV(Oi) -> Zi \n'.format('SIGMOID_INV'))
		instr += encode_instr(OPCodes.ACT, r, ACTType.SIGMOID_INV)

		f.write('ACT {} #SIGMOID_DER(Zi) \n'.format('SIGMOID_DER'))
		instr += encode_instr(OPCodes.ACT, r, ACTType.SIGMOID_DER)

		f.write('ALU {} {} #Oi_bp*SIGMOID_DER(Zi)->Zi_bp \n'.format('Mult', r))
		instr += encode_instr(OPCodes.MULT, r)

		f.write('DRAM_WR {} {} #Write back oi_bp\n'.format(mem_chunk, oi_bp_id))
		instr += encode_instr(OPCodes.DRAM_WR, mem_chunk, oi_bp_id)



		f.write('ALU {} {} #tmp_res*Cin \n'.format('Mult', r))
		instr += encode_instr(OPCodes.MULT, r)	

		f.write('ACT {} #SIGMOID_INV(Of) -> Zf \n'.format('SIGMOID_INV'))
		instr += encode_instr(OPCodes.ACT, r, ACTType.SIGMOID_INV)

		f.write('ACT {} #SIGMOID_DER(Zf) \n'.format('SIGMOID_DER'))
		instr += encode_instr(OPCodes.ACT, r, ACTType.SIGMOID_DER)

		f.write('ALU {} {} #Of_bp*SIGMOID_DER(Zf)->Zf_bp \n'.format('Mult', r))
		instr += encode_instr(OPCodes.MULT, r)

		f.write('DRAM_WR {} {} #Write back of_bp\n'.format(mem_chunk, of_bp_id))
		instr += encode_instr(OPCodes.DRAM_WR, mem_chunk, of_bp_id)


		f.write('ALU {} {} #tmp_res*of \n'.format('Mult', r))
		instr += encode_instr(OPCodes.MULT, r)	

		f.write('DRAM_WR {} {} #Write back Cin_bp\n'.format(mem_chunk, Cin_bp_id))
		instr += encode_instr(OPCodes.DRAM_WR, mem_chunk, Cin_bp_id)


		mem_cnt += mem_chunk

	f.write('EOL #End of layer\n')
	instr += encode_instr(OPCodes.EOL)

	return instr

def eltwise_instr(instr, dir_name, name, tensor_ids, OH, OW, OC, B, acc_obj, vector_op=None, activation=None):
	data1_id, data2_id, out_id = tensor_ids

	N = acc_obj.N
	M = acc_obj.M
	oprec = acc_obj.prec[3]

	input_output_size = OH*OW*OC*B*oprec
	
	r=100 #large r means larger data chunks, less instructions.
	mem_chunk = N*M*oprec*r
	
	f= open(dir_name+name+".txt","w")

	mem_cnt = 0
	while mem_cnt < input_output_size:
		f.write('DRAM_RD {} {} #Read vec1\n'.format(mem_chunk, data1_id))
		instr += encode_instr(OPCodes.DRAM_RD, mem_chunk, data1_id)
		f.write('DRAM_RD {} {} #Read vec2\n'.format(mem_chunk, data2_id))
		instr += encode_instr(OPCodes.DRAM_RD, mem_chunk, data2_id)


		f.write('ALU {} {} #Vector op \n'.format(vector_op, r))
		if vector_op == 'Add':
			instr += encode_instr(OPCodes.ADD, r)
		elif vector_op == 'Mult':
			instr += encode_instr(OPCodes.MULT, r)
		elif vector_op == 'Mult_der':
			instr += encode_instr(OPCodes.MULT_DER, r)
		else:
			assert False, 'ALU op is not defined'

		if activation is not None:
			f.write('ACT {} #Apply activation function \n'.format(activation))
			instr += encode_instr(OPCodes.ACT, r, activation)

		f.write('DRAM_WR {} {} #Write back output\n'.format(mem_chunk, out_id))
		instr += encode_instr(OPCodes.DRAM_WR, mem_chunk, out_id)
		mem_cnt += mem_chunk

	f.write('EOL #End of layer\n')
	instr += encode_instr(OPCodes.EOL)

	return instr

def maxpool2instr(instr, dir_name, name, tensor_ids, OH, OW, OC, B, acc_obj, pool_kernel=None):
	inp_id, out_id = tensor_ids

	N = acc_obj.N
	M = acc_obj.M
	oprec = acc_obj.prec[3]

	total_output_size = OH*OW*OC*B

	r=10 #large r means larger data chunks, less instructions.
	mem_chunk_in = N*M*pool_kernel[0]*pool_kernel[1]*oprec*r
	mem_chunk_out = N*M*oprec*r

	f= open(dir_name+name+".txt","w")

	mem_cnt = 0
	while mem_cnt < total_output_size:
		f.write('DRAM_RD {} {} #Read Input\n'.format(mem_chunk_in, inp_id))
		instr += encode_instr(OPCodes.DRAM_RD, mem_chunk_in, inp_id)

		f.write('POOL {}_{} #Pooling \n'.format(pool_kernel[0], pool_kernel[1]))
		instr += encode_instr(OPCodes.POOL, r, pool_kernel[0])

		f.write('DRAM_WR {} {} #Write output\n'.format(mem_chunk_out, out_id))
		instr += encode_instr(OPCodes.DRAM_WR, mem_chunk_out, out_id)
		mem_cnt += N*M*r

	f.write('EOL #End of layer\n')
	instr += encode_instr(OPCodes.EOL)

	return instr

def maxpool2instr_bp(instr, dir_name, name, tensor_ids, OH, OW, OC, B, acc_obj, pool_kernel=None):
	inp_id, out_id = tensor_ids

	N = acc_obj.N
	M = acc_obj.M
	oprec = acc_obj.prec[3]

	total_output_size = OH*OW*OC*B

	mem_chunk_in = N*M*oprec
	mem_chunk_out = N*M*pool_kernel[0]*pool_kernel[1]*oprec

	f= open(dir_name+name+".txt","w")

	mem_cnt = 0
	while mem_cnt < total_output_size:
		f.write('DRAM_RD {} {} #Read Input\n'.format(mem_chunk_in, out_id))
		instr += encode_instr(OPCodes.DRAM_RD, mem_chunk_in, out_id)

		f.write('POOL_BP {} {} #BP pooling \n'.format(pool_kernel[0], pool_kernel[1]))
		instr += encode_instr(OPCodes.POOL_BP, 1, pool_kernel[0])

		f.write('DRAM_WR {} {} #Write output\n'.format(mem_chunk_out, inp_id))
		instr += encode_instr(OPCodes.DRAM_WR, mem_chunk_out, inp_id)
		mem_cnt += N*M

	f.write('EOL #End of layer\n')
	instr += encode_instr(OPCodes.EOL)
	
	return instr

#TODO: Agree on with Mario how instructions will work for bnorm
def bnorm2instr(instr, dir_name, name, tensor_ids, OH, OW, OC, B, acc_obj, activation=None):
	z_id, gamma_id, beta_id, out_id = tensor_ids

	N = acc_obj.N
	M = acc_obj.M
	oprec = acc_obj.prec[3]

	total_output_size = OH*OW*OC*B*oprec
	
	mem_chunk = OH*OW*B*oprec

	f= open(dir_name+name+".txt","w")

	mem_cnt = 0
	while mem_cnt < total_output_size:
		f.write('DRAM_RD {} {} #Read z\n'.format(mem_chunk, z_id))
		instr += encode_instr(OPCodes.DRAM_RD, mem_chunk, z_id)
		f.write('DRAM_RD {} {} #Read gamma\n'.format(oprec, gamma_id))
		instr += encode_instr(OPCodes.DRAM_RD, oprec, gamma_id)
		f.write('DRAM_RD {} {} #Read beta\n'.format(oprec, beta_id))
		instr += encode_instr(OPCodes.DRAM_RD, oprec, beta_id)


		f.write('BNORM {} #Batch norm \n'.format(B))
		instr += encode_instr(OPCodes.BNORM, 1, B)
		if activation is not None:
			f.write('ACT {} #Apply activation function \n'.format(activation))
			instr += encode_instr(OPCodes.ACT, 1, activation)


		f.write('DRAM_WR {} {} #Write output\n'.format(mem_chunk, out_id))
		instr += encode_instr(OPCodes.DRAM_WR, mem_chunk, out_id)
		mem_cnt += mem_chunk

	f.write('EOL #End of layer\n')
	instr += encode_instr(OPCodes.EOL)

	return instr

# According to http://cthorey.github.io./backpropagation/
def bnorm2instr_bp(instr,dir_name,name, tensor_ids, OH, OW, OC, B, acc_obj):
	h_id, dh_id, gamma_id, dgamma_id, beta_id, dbeta_id, dy_id = tensor_ids

	N = acc_obj.N
	M = acc_obj.M
	oprec = acc_obj.prec[3]

	total_output_size = OH*OW*OC*B*oprec
	
	mem_chunk = OH*OW*B*oprec

	f= open(dir_name+name+".txt","w")

	mem_cnt = 0
	while mem_cnt < total_output_size:
		f.write('DRAM_RD {} #Read h\n'.format(mem_chunk))
		instr += encode_instr(OPCodes.DRAM_RD, mem_chunk, h_id)
		f.write('DRAM_RD {} #Read dy\n'.format(mem_chunk))
		instr += encode_instr(OPCodes.DRAM_RD, mem_chunk, dy_id)
		f.write('DRAM_RD {} #Read gamma\n'.format(oprec))
		instr += encode_instr(OPCodes.DRAM_RD, oprec, gamma_id)
		f.write('DRAM_RD {} #Read beta\n'.format(oprec))
		instr += encode_instr(OPCodes.DRAM_RD, oprec, beta_id)

		f.write('BNORM_BP {} {} #Calc. dh, dgamma, dbeta \n'.format(B, 1))
		instr += encode_instr(OPCodes.BNORM_BP, 1, B)

		f.write('DRAM_WR {} #Write dh\n'.format(mem_chunk))
		instr += encode_instr(OPCodes.DRAM_WR, mem_chunk, dh_id)
		f.write('DRAM_WR {} #Write dgamma\n'.format(oprec))
		instr += encode_instr(OPCodes.DRAM_WR, oprec, dgamma_id)
		f.write('DRAM_WR {} #Write dbeta\n'.format(oprec))
		instr += encode_instr(OPCodes.DRAM_WR, oprec, dbeta_id)

		mem_cnt += mem_chunk

	f.write('EOL #End of layer\n')
	instr += encode_instr(OPCodes.EOL)
	return instr
