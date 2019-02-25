import math
import logging
import numpy as np

from dnnweaver2.compiler import GraphCompiler
from dnnweaver2.compiler.bp_compiler import BP_GraphCompiler, MacroNode
from dnnweaver2.benchmarks import get_graph
from dnnweaver2.graph import Graph
from dnnweaver2 import get_tensor
from dnnweaver2.scalar.dtypes import FQDtype, FixedPoint
from dnnweaver2.tensorOps.cnn import conv2D,conv2D_bp, maxPool, Convolution, flatten, matmul

from custom_isa import OPCodes, ACTType, MACType

tile_deps = {}
tile_deps['B/b']   = {'ibuf': True,  'wbuf': False, 'obuf': True,  'bbuf': False}
tile_deps['OW/ow'] = {'ibuf': True,  'wbuf': False, 'obuf': True,  'bbuf': False}
tile_deps['OH/oh'] = {'ibuf': True,  'wbuf': False, 'obuf': True,  'bbuf': False}
tile_deps['IC/ic'] = {'ibuf': True,  'wbuf': True,  'obuf': False, 'bbuf': False}
tile_deps['OC/oc'] = {'ibuf': False, 'wbuf': True,  'obuf': True,  'bbuf': True}

tile_deps_bp = {}
tile_deps_bp['B/b']   = {'ibuf': True,  'wbuf': False, 'obuf': True}
tile_deps_bp['OW/ow'] = {'ibuf': True,  'wbuf': False, 'obuf': True}
tile_deps_bp['OH/oh'] = {'ibuf': True,  'wbuf': False, 'obuf': True}
tile_deps_bp['IC/ic'] = {'ibuf': True,  'wbuf': True,  'obuf': False}
tile_deps_bp['OC/oc'] = {'ibuf': False, 'wbuf': True,  'obuf': True}


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


def conv2instr(instr, stats, dir_name, layer_name, tensor_ids, OH, OW, OC, IC, B, K, S, acc_obj, mac_type=MACType.CONV, activation=None, transpose_weight=False):
	IH = OH*S
	IW = OW*S

	data_id, weight_id, bias_id, out_id = tensor_ids

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

	obuf_fetch = False
	ibuf_fetch = True
	wbuf_fetch = True
	bbuf_fetch = True

	retire_output = False

	cycle = 0
	dram_read = 0
	dram_write = 0

	out_tiles_access = np.zeros((tiling['OH/oh'][0],tiling['OW/ow'][0],tiling['OC/oc'][0],tiling['B/b'][0]))

	f= open(dir_name+layer_name+".txt","w")

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
							dram_write += output_tile_size
							f.write('DRAM_WR {} {} #Write back output tile\n'.format(output_tile_size, out_id))
							instr += encode_instr(OPCodes.DRAM_WR,output_tile_size,out_id)

							cycle += int(math.ceil(float(output_tile_size) / DRAM_BW))

							if out_tiles_access[oh_ind,ow_ind,oc_ind,b_ind]>0: #Skip reading output tile if it is first time
								dram_read += output_tile_size
								f.write('DRAM_RD {} #Read output tile\n'.format(output_tile_size, out_id))
								instr += encode_instr(OPCodes.DRAM_RD,output_tile_size,out_id)

								cycle += int(math.ceil(float(output_tile_size) / DRAM_BW))
							out_tiles_access[oh_ind,ow_ind,oc_ind,b_ind] = out_tiles_access[oh_ind,ow_ind,oc_ind,b_ind] + 1

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
								f.write('DRAM_RD_TP {} {} {} #Read weight tile\n'.format(weight_tile_size, weight_id, K))
								instr += encode_instr(OPCodes.DRAM_RD_TP, weight_tile_size, weight_id, K)
							else:
								f.write('DRAM_RD {} {} #Read weight tile\n'.format(weight_tile_size, weight_id))
								instr += encode_instr(OPCodes.DRAM_RD,weight_tile_size,weight_id)
							cycle += int(math.ceil(float(weight_tile_size) / DRAM_BW))
							wbuf_fetch = False

						if bbuf_fetch == True:
							dram_read += bias_tile_size
							f.write('DRAM_RD {} {} #Read bias tile\n'.format(bias_tile_size, bias_id))
							instr += encode_instr(OPCodes.DRAM_RD,bias_tile_size,bias_id)

							cycle += int(math.ceil(float(bias_tile_size) / DRAM_BW))
							bbuf_fetch = False
						



						
						compute_cycles_per_pe = math.ceil(float(_ic)/num_rows) * math.ceil(float(_oc)/num_cols) * kw * kh * _oh * _ow * _b 
						f.write('ALU {} #Compute inner loop \n'.format(int(compute_cycles_per_pe)))
						instr += encode_instr(OPCodes.MAC, int(compute_cycles_per_pe), mac_type)

						if retire_output==True:
							if activation is not None:
								f.write('ACT {}_{} {} #Apply activation function \n'.format(activation))
								instr += encode_instr(OPCodes.ACT, math.ceil(float(_oc)/num_cols), activation)




						cycle += math.ceil(float(_ic)/num_rows) * math.ceil(float(_oc)/num_cols) * kw * kh * _oh * _ow * _b 
						if retire_output==True:
							cycle += 2 * math.ceil(float(_oc)/num_cols) * _oh * _ow * _b 
							retire_output = False

						if tile_deps[order[0]]['obuf'] == True:
							obuf_fetch = True
						if tile_deps[order[0]]['ibuf'] == True:
							ibuf_fetch = True
						if tile_deps[order[0]]['wbuf'] == True:
							wbuf_fetch = True
						if tile_deps[order[0]]['bbuf'] == True:
							bbuf_fetch = True

					if tile_deps[order[1]]['obuf'] == True:
						obuf_fetch = True
					if tile_deps[order[1]]['ibuf'] == True:
						ibuf_fetch = True
					if tile_deps[order[1]]['wbuf'] == True:
						wbuf_fetch = True
					if tile_deps[order[1]]['bbuf'] == True:
						bbuf_fetch = True

				if tile_deps[order[2]]['obuf'] == True:
					obuf_fetch = True
				if tile_deps[order[2]]['ibuf'] == True:
					ibuf_fetch = True
				if tile_deps[order[2]]['wbuf'] == True:
					wbuf_fetch = True
				if tile_deps[order[2]]['bbuf'] == True:
					bbuf_fetch = True

			if tile_deps[order[3]]['obuf'] == True:
				obuf_fetch = True
			if tile_deps[order[3]]['ibuf'] == True:
				ibuf_fetch = True
			if tile_deps[order[3]]['wbuf'] == True:
				wbuf_fetch = True
			if tile_deps[order[3]]['bbuf'] == True:
				bbuf_fetch = True

		if tile_deps[order[4]]['obuf'] == True:
			obuf_fetch = True
		if tile_deps[order[4]]['ibuf'] == True:
			ibuf_fetch = True
		if tile_deps[order[4]]['wbuf'] == True:
			wbuf_fetch = True
		if tile_deps[order[4]]['bbuf'] == True:
			bbuf_fetch = True

	dram_write += output_tile_size
	f.write('DRAM_WR {} {} #Write back output tile\n'.format(output_tile_size, out_id))
	instr += encode_instr(OPCodes.DRAM_WR,output_tile_size,out_id)

	cycle += int(math.ceil(float(output_tile_size) / DRAM_BW))

	f.write('EOL #End of layer\n')
	instr += encode_instr(OPCodes.EOL)
	f.close()

	print('Total DRAM read simulated: {:,}'.format(dram_read))
	print('Total DRAM write simulated: {:,}'.format(dram_write))
	print('Total cycles simulated: {:,}'.format(cycle))

	print('Memory cycles: {:,}'.format(stats.memory_cycles_required))
	print('Compute cycles: {:,}'.format(stats.compute_cycles))

	return instr

# https://www.jefkine.com/general/2016/09/05/backpropagation-in-convolutional-neural-networks/
# http://neuralnetworksanddeeplearning.com/chap2.html
def conv2instr_bp(instr, stats, dir_name, layer_name, tensor_ids, OH, OW, OC, IC, B, K, S, acc_obj, mac_type=MACType.CONV, activation=None):
	IH = OH*S
	IW = OW*S

	out_id, data_id, _, weight_id = tensor_ids

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

	obuf_fetch = False
	ibuf_fetch = True
	wbuf_fetch = True
	bbuf_fetch = True

	retire_output = False

	cycle = 0
	dram_read = 0
	dram_write = 0

	out_tiles_access = np.zeros((tiling['IC/ic'][0],tiling['OC/oc'][0]))

	f= open(dir_name+layer_name+".txt","w")

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
						
						if oh_ind == tiling['OH/oh'][0]-1 and ow_ind == tiling['OW/ow'][0]-1 and b_ind == tiling['B/b'][0]-1:
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
						weight_tile_size = kw*kh*_oc*_ic*wprec
						output_tile_size = _oh*_ow*_oc*_b*oprec

						if wbuf_fetch == True:
							dram_write += weight_tile_size
							f.write('DRAM_WR {} {} #Write back output tile\n'.format(weight_tile_size, weight_id))
							instr += encode_instr(OPCodes.DRAM_WR, weight_tile_size, weight_id)
							cycle += int(math.ceil(float(weight_tile_size) / DRAM_BW))

							if out_tiles_access[ic_ind,oc_ind]>0: #Skip reading output tile if it is first time
								dram_read += weight_tile_size
								f.write('DRAM_RD {} {} #Read output tile\n'.format(weight_tile_size, weight_id))
								instr += encode_instr(OPCodes.DRAM_RD, weight_tile_size, weight_id)
								cycle += int(math.ceil(float(weight_tile_size) / DRAM_BW))
							out_tiles_access[ic_ind,oc_ind] += 1

							wbuf_fetch = False

						if ibuf_fetch == True:
							dram_read += data_tile_size
							f.write('DRAM_RD {} {} #Read input tile\n'.format(data_tile_size, data_id))
							instr += encode_instr(OPCodes.DRAM_RD, data_tile_size, data_id)
							cycle += int(math.ceil(float(data_tile_size) / DRAM_BW))
							ibuf_fetch = False

						if obuf_fetch == True:
							dram_read += output_tile_size
							f.write('DRAM_RD {} {} #Read weight tile\n'.format(output_tile_size, out_id))
							instr += encode_instr(OPCodes.DRAM_RD, output_tile_size, out_id)
							cycle += int(math.ceil(float(output_tile_size) / DRAM_BW))
							obuf_fetch = False


						compute_cycles_per_pe = math.ceil(float(_ic)/num_rows) * math.ceil(float(_oc)/num_cols) * kw * kh * _oh * _ow * _b
						f.write('ALU {} #Compute inner loop \n'.format(int(compute_cycles_per_pe)))
						instr += encode_instr(OPCodes.MAC, int(compute_cycles_per_pe), mac_type)

						cycle += math.ceil(float(_ic)/num_rows) * math.ceil(float(_oc)/num_cols) * kw * kh * _oh * _ow * _b
						if retire_output==True:
							retire_output = False

						if tile_deps_bp[order[0]]['obuf'] == True:
							obuf_fetch = True
						if tile_deps_bp[order[0]]['ibuf'] == True:
							ibuf_fetch = True
						if tile_deps_bp[order[0]]['wbuf'] == True:
							wbuf_fetch = True

					if tile_deps_bp[order[1]]['obuf'] == True:
						obuf_fetch = True
					if tile_deps_bp[order[1]]['ibuf'] == True:
						ibuf_fetch = True
					if tile_deps_bp[order[1]]['wbuf'] == True:
						wbuf_fetch = True

				if tile_deps_bp[order[2]]['obuf'] == True:
					obuf_fetch = True
				if tile_deps_bp[order[2]]['ibuf'] == True:
					ibuf_fetch = True
				if tile_deps_bp[order[2]]['wbuf'] == True:
					wbuf_fetch = True

			if tile_deps_bp[order[3]]['obuf'] == True:
				obuf_fetch = True
			if tile_deps_bp[order[3]]['ibuf'] == True:
				ibuf_fetch = True
			if tile_deps_bp[order[3]]['wbuf'] == True:
				wbuf_fetch = True

		if tile_deps_bp[order[4]]['obuf'] == True:
			obuf_fetch = True
		if tile_deps_bp[order[4]]['ibuf'] == True:
			ibuf_fetch = True
		if tile_deps_bp[order[4]]['wbuf'] == True:
			wbuf_fetch = True

	dram_write += weight_tile_size
	f.write('DRAM_WR {} {} #Write back output tile\n'.format(weight_tile_size, weight_id))
	instr += encode_instr(OPCodes.DRAM_WR, weight_tile_size, weight_id)
	cycle += int(math.ceil(float(weight_tile_size) / DRAM_BW))

	f.write('EOL #End of layer\n')
	instr += encode_instr(OPCodes.EOL)
	f.close()

	print('Total DRAM read simulated: {:,}'.format(dram_read))
	print('Total DRAM write simulated: {:,}'.format(dram_write))
	print('Total cycles simulated: {:,}'.format(cycle))

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

	total_input_size = OH*OW*OC*B*oprec
	
	mem_chunk = OH*OW*B*oprec

	f= open(dir_name+name+".txt","w")

	mem_cnt = 0
	while mem_cnt < total_input_size:
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

#TODO: Update this for tensor indexes
# According to http://cthorey.github.io./backpropagation/
def bnorm2instr_bp(instr,dir_name,name, tensor_ids,OH, OW, OC, B, acc_obj):
	N = acc_obj.N
	M = acc_obj.M
	oprec = acc_obj.prec[3]

	total_output_size = OH*OW*OC*B*oprec
	
	mem_chunk = OH*OW*B*oprec

	f= open(dir_name+name+".txt","w")

	mem_cnt = 0
	while mem_cnt < total_output_size:
		f.write('DRAM_RD {} #Read h\n'.format(mem_chunk))
		f.write('DRAM_RD {} #Read dy\n'.format(mem_chunk))
		f.write('DRAM_RD {} #Read gamma\n'.format(mem_chunk/B))
		f.write('DRAM_RD {} #Read beta\n'.format(mem_chunk/B))

		for i in range(N):
			for j in range(M):
				f.write('BNORM_BP {}_{} {} {} #Calc. dh, dgamma, dbeta \n'.format(i, j, B, 1))

		f.write('DRAM_WR {} #Write dh\n'.format(mem_chunk))
		f.write('DRAM_WR {} #Write dgamma\n'.format(mem_chunk/B))
		f.write('DRAM_WR {} #Write dbeta\n'.format(mem_chunk/B))

		mem_cnt += mem_chunk

	f.write('EOL #End of layer\n')
	return instr
