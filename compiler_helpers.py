import math
import logging
import numpy as np

from dnnweaver2.compiler import GraphCompiler,MacroNode
from dnnweaver2.benchmarks import get_graph
from dnnweaver2.graph import Graph
from dnnweaver2 import get_tensor
from dnnweaver2.scalar.dtypes import FQDtype, FixedPoint
from dnnweaver2.tensorOps.cnn import conv2D, maxPool, Convolution, flatten, matmul


tile_deps = {}
tile_deps['B/b']   = {'ibuf': True,  'wbuf': False, 'obuf': True,  'bbuf': False}
tile_deps['OW/ow'] = {'ibuf': True,  'wbuf': False, 'obuf': True,  'bbuf': False}
tile_deps['OH/oh'] = {'ibuf': True,  'wbuf': False, 'obuf': True,  'bbuf': False}
tile_deps['IC/ic'] = {'ibuf': True,  'wbuf': True,  'obuf': False, 'bbuf': False}
tile_deps['OC/oc'] = {'ibuf': False, 'wbuf': True,  'obuf': True,  'bbuf': True}


def conv(IH, IW, IC, B, K, OC, S, acc_obj):
	graph = Graph('resnet50', dataset='imagenet', log_level=logging.ERROR)

	with graph.as_default():

	    with graph.name_scope('inputs'):
		i = get_tensor(shape=(B,IH,IW,IC), name='data', dtype=FQDtype.FXP16, trainable=False)

	    with graph.name_scope('conv0'):
		weights = get_tensor(shape=(OC, K, K, IC),
		                     name='weights',
		                     dtype=FixedPoint(16,12))
		biases = get_tensor(shape=(OC),
		                     name='biases',
		                     dtype=FixedPoint(32,20))
		conv = conv2D(i, weights, biases, stride=(1,S,S,1), pad='SAME', dtype=FixedPoint(16,8))

	compiler = GraphCompiler(log_level=logging.ERROR)
	inst_binary, stats = compiler.compile(graph=graph, acc_obj=acc_obj)

	return stats

def fc(B, IC, OC, acc_obj):
	graph = Graph('resnet50', dataset='imagenet', log_level=logging.ERROR)

	with graph.as_default():

	    with graph.name_scope('inputs'):
		i = get_tensor(shape=(B,1,1,IC), name='data', dtype=FQDtype.FXP16, trainable=False)

	    with graph.name_scope('conv0'):
		weights = get_tensor(shape=(OC, 1, 1, IC),
		                     name='weights',
		                     dtype=FixedPoint(16,12))
		biases = get_tensor(shape=(OC),
		                     name='biases',
		                     dtype=FixedPoint(32,20))
		conv = conv2D(i, weights, biases, stride=(1,1,1,1), pad='SAME', dtype=FixedPoint(16,8))

	compiler = GraphCompiler(log_level=logging.ERROR)
	inst_binary, stats = compiler.compile(graph=graph, acc_obj=acc_obj)

	return stats

def conv_instr_writer(stats, layer_name, IH, IW, IC, B, K, OC, S, num_rows, num_cols, DRAM_BW, iprec, wprec, bprec, oprec, activation=None, pooling=False, pool_window=None, batch_norm=False):
	OH = IH/S
	OW = IW/S

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
	reduce_output_size = False

	cycle = 0
	dram_read = 0
	dram_write = 0

	out_tiles_access = np.zeros((tiling['OH/oh'][0],tiling['OW/ow'][0],tiling['OC/oc'][0],tiling['B/b'][0]))

	f= open("resnet50/"+layer_name+".txt","w")

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
							if reduce_output_size == True:
								output_tile_size = output_tile_size / (pool_window[0] * pool_window[1])
								reduce_output_size = False


							dram_write += output_tile_size
							f.write('DRAM_WR {} #Write back output tile\n'.format(output_tile_size))
							cycle += int(math.ceil(float(output_tile_size) / DRAM_BW))

							if out_tiles_access[oh_ind,ow_ind,oc_ind,b_ind]>0: #Skip reading output tile if it is first time
								dram_read += output_tile_size
								f.write('DRAM_RD {} #Read output tile\n'.format(output_tile_size))
								cycle += int(math.ceil(float(output_tile_size) / DRAM_BW))
							out_tiles_access[oh_ind,ow_ind,oc_ind,b_ind] = out_tiles_access[oh_ind,ow_ind,oc_ind,b_ind] + 1

							obuf_fetch = False

						if ibuf_fetch == True:
							dram_read += data_tile_size
							f.write('DRAM_RD {} #Read input tile\n'.format(data_tile_size))
							cycle += int(math.ceil(float(data_tile_size) / DRAM_BW))
							ibuf_fetch = False

						if wbuf_fetch == True:
							dram_read += weight_tile_size
							f.write('DRAM_RD {} #Read weight tile\n'.format(weight_tile_size))
							cycle += int(math.ceil(float(weight_tile_size) / DRAM_BW))
							wbuf_fetch = False

						if bbuf_fetch == True:
							dram_read += bias_tile_size
							f.write('DRAM_RD {} #Read bias tile\n'.format(bias_tile_size))
							cycle += int(math.ceil(float(bias_tile_size) / DRAM_BW))
							bbuf_fetch = False
						
						first_pe_load_x = math.ceil(float(_ic)/num_rows)
						first_pe_load_y = math.ceil(float(_oc)/num_cols)
						last_pe_load_x = _ic-(num_rows-1)*first_pe_load_x
						last_pe_load_y = _oc-(num_cols-1)*first_pe_load_y

						for pe_x in range(num_rows-1):
							for pe_y in range(num_cols-1):
								compute_cycles_per_pe = first_pe_load_x * first_pe_load_y * kw * kh * _oh * _ow * _b 
								f.write('ALU {}_{} {} #Compute inner loop \n'.format(pe_x, pe_y, int(compute_cycles_per_pe)))
								if retire_output==True:
									if batch_norm==True:
										f.write('BNORM {}_{} #Batch normalization \n'.format(pe_x, pe_y))
									if activation is not None:
										f.write('ACT {}_{} {} #Apply activation function \n'.format(pe_x, pe_y, activation))
									if pooling==True:
										f.write('POOL {}_{} #Apply pooling function \n'.format(pe_x, pe_y))

							compute_cycles_per_pe = first_pe_load_x * last_pe_load_y * kw * kh * _oh * _ow * _b

							f.write('ALU {}_{} {} #Compute inner loop \n'.format(pe_x, num_cols-1, int(compute_cycles_per_pe)))
							if retire_output==True and compute_cycles_per_pe>0:
								if batch_norm==True:
									f.write('BNORM {}_{} #Batch normalization \n'.format(pe_x, num_cols-1))
								if activation is not None:
									f.write('ACT {}_{} {} #Apply activation function \n'.format(pe_x, num_cols-1, activation))
								if pooling==True:
									f.write('POOL {}_{} #Apply pooling function \n'.format(pe_x, num_cols-1))
						
						for pe_y in range(num_cols-1):
							compute_cycles_per_pe = last_pe_load_x * first_pe_load_y * kw * kh * _oh * _ow * _b 

							f.write('ALU {}_{} {} #Compute inner loop \n'.format(num_rows-1, pe_y, int(compute_cycles_per_pe)))
							if retire_output==True and compute_cycles_per_pe>0:
								if batch_norm==True:
									f.write('BNORM {}_{} #Batch normalization \n'.format(num_rows-1, pe_y))
								if activation is not None:
									f.write('ACT {}_{} {} #Apply activation function \n'.format(num_rows-1, pe_y, activation))
								if pooling==True:
									f.write('POOL {}_{} #Apply pooling function \n'.format(num_rows-1, pe_y))

						compute_cycles_per_pe = last_pe_load_x * last_pe_load_y * kw * kh * _oh * _ow * _b 

						f.write('ALU {}_{} {} #Compute inner loop \n'.format(num_rows-1, num_cols-1, int(compute_cycles_per_pe)))
						if retire_output==True and compute_cycles_per_pe>0:
							if batch_norm==True:
								f.write('BNORM {}_{} #Batch normalization \n'.format(num_rows-1, num_cols-1))
							if activation is not None:
								f.write('ACT {}_{} {} #Apply activation function \n'.format(num_rows-1, num_cols-1, activation))
							if pooling==True:
								f.write('POOL {}_{} #Apply pooling function \n'.format(num_rows-1, num_cols-1))

						cycle += first_pe_load_x * first_pe_load_y * kw * kh * _oh * _ow * _b 
						if retire_output==True:
							cycle += 2 * first_pe_load_y * _oh * _ow * _b 
							if pooling==True:
								cycle += 4 * first_pe_load_y * _oh * _ow * _b
								reduce_output_size = True
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


	if reduce_output_size == True:
		output_tile_size = output_tile_size / (pool_window[0] * pool_window[1])
		reduce_output_size = False	
	
	dram_write += output_tile_size
	f.write('DRAM_WR {} #Write back output tile\n'.format(output_tile_size))
	cycle += int(math.ceil(float(output_tile_size) / DRAM_BW))

	f.write('EOL #End of layer\n')
	f.close()

	print('Total DRAM read simulated: {:,}'.format(dram_read))
	print('Total DRAM write simulated: {:,}'.format(dram_write))
	print('Total cycles simulated: {:,}'.format(cycle))

	print('***********************************************************')
	print('DRAM reads: {:,}'.format(stats.reads['dram']))
	print('DRAM writes: {:,}'.format(stats.writes['dram']))
	print('Initial DRAM reads: {}'.format(stats.initial_dram_reads))
	print('Final DRAM writes: {}'.format(stats.final_dram_writes))
	print('Total DRAM access: {}'.format(stats.total_dram_accesses))
	print('Middle DRAM access: {}'.format(stats.middle_dram_accesses))
	print('SRAM reads: IBUF:{} WBUF:{} OBUF:{}'.format(stats.reads['ibuf'],stats.reads['wbuf'],stats.reads['obuf']))
	print('Total compute cycles:{}'.format(stats.compute_cycles))
	print('Total memory cycles:{}'.format(stats.memory_cycles_required))
	print('Memory stall cycles:{}'.format(stats.mem_stall_cycles))
	print('Total cycles:{}'.format(stats.total_cycles))


def fc_instr_writer(stats, layer_name, IC, B, OC, num_rows, num_cols, DRAM_BW, iprec, wprec, bprec, oprec, activation=None, batch_norm=False):
	tiling = stats.tiling
	order = stats.order

	print('Order and tiling:')
	for o in order:
		print('{}: {}-{}'.format(o,tiling[o][0],tiling[o][1]))

	ic = tiling['IC/ic'][1]
	oc = tiling['OC/oc'][1]
	b = tiling['B/b'][1]

	obuf_fetch = False
	ibuf_fetch = True
	wbuf_fetch = True
	bbuf_fetch = True

	retire_output = False

	out_tiles_access = np.zeros((tiling['OC/oc'][0],tiling['B/b'][0]))

	cycle = 0
	dram_read = 0
	dram_write = 0

	f= open("resnet50/"+layer_name+".txt","w")

	for i in range(tiling[order[2]][0]):
		if order[2]=='OC/oc':
			oc_ind = i
		if order[2]=='B/b':
			b_ind = i
		if order[2]=='IC/ic':
			ic_ind = i

		for j in range(tiling[order[1]][0]):
			if order[1]=='OC/oc':
				oc_ind = j
			if order[1]=='B/b':
				b_ind = j
			if order[1]=='IC/ic':
				ic_ind = j

			for k in range(tiling[order[0]][0]):
				if order[0]=='OC/oc':
					oc_ind = k
				if order[0]=='B/b':
					b_ind = k
				if order[0]=='IC/ic':
					ic_ind = k

				if ic_ind == tiling['IC/ic'][0]-1:
					retire_output = True

				_oc = min(oc,OC-oc_ind*oc)
				_b = min(b,B-b_ind*b)
				_ic = min(ic,IC-ic_ind*ic)

				data_tile_size = _ic*_b*iprec
				weight_tile_size = _ic*_oc*wprec
				bias_tile_size = _oc*bprec
				output_tile_size = _oc*_b*oprec

				if obuf_fetch == True:
					dram_write += output_tile_size
					f.write('DRAM_WR {} #Write back output tile\n'.format(output_tile_size))
					cycle += int(math.ceil(float(output_tile_size) / DRAM_BW))

					if out_tiles_access[oc_ind,b_ind]>0: #Skip reading output tile if it is first time
						dram_read += output_tile_size
						f.write('DRAM_RD {} #Read output tile\n'.format(output_tile_size))
						cycle += int(math.ceil(float(output_tile_size) / DRAM_BW))
					out_tiles_access[oc_ind,b_ind] = out_tiles_access[oc_ind,b_ind] + 1

					obuf_fetch = False

				if ibuf_fetch == True:
					dram_read += data_tile_size
					f.write('DRAM_RD {} #Read input tile\n'.format(data_tile_size))
					cycle += int(math.ceil(float(data_tile_size) / DRAM_BW))
					ibuf_fetch = False

				if wbuf_fetch == True:
					dram_read += weight_tile_size
					f.write('DRAM_RD {} #Read weight tile\n'.format(weight_tile_size))
					cycle += int(math.ceil(float(weight_tile_size) / DRAM_BW))
					wbuf_fetch = False

				if bbuf_fetch == True:
					dram_read += bias_tile_size
					f.write('DRAM_RD {} #Read bias tile\n'.format(bias_tile_size))
					cycle += int(math.ceil(float(bias_tile_size) / DRAM_BW))
					bbuf_fetch = False


				first_pe_load_x = math.ceil(float(_ic)/num_rows)
				first_pe_load_y = math.ceil(float(_oc)/num_cols)
				last_pe_load_x = _ic-(num_rows-1)*first_pe_load_x
				last_pe_load_y = _oc-(num_cols-1)*first_pe_load_y

				for pe_x in range(num_rows-1):
					for pe_y in range(num_cols-1):
						compute_cycles_per_pe = first_pe_load_x * first_pe_load_y * _b 
						f.write('ALU {}_{} {} #Compute inner loop \n'.format(pe_x, pe_y, int(compute_cycles_per_pe)))
						if retire_output==True:
							if batch_norm==True:
								f.write('BNORM {}_{} #Batch normalization \n'.format(pe_x, pe_y))
							if activation is not None:
								f.write('ACT {}_{} {} #Apply activation function \n'.format(pe_x, pe_y, activation))

					compute_cycles_per_pe = first_pe_load_x * last_pe_load_y * _b
					f.write('ALU {}_{} {} #Compute inner loop \n'.format(pe_x, num_cols-1, int(compute_cycles_per_pe)))
					if retire_output==True and compute_cycles_per_pe>0:
						if batch_norm==True:
							f.write('BNORM {}_{} #Batch normalization \n'.format(pe_x, num_cols-1))
						if activation is not None:
							f.write('ACT {}_{} {} #Apply activation function \n'.format(pe_x, num_cols-1, activation))
				
				for pe_y in range(num_cols-1):
					compute_cycles_per_pe = last_pe_load_x * first_pe_load_y * _b 

					f.write('ALU {}_{} {} #Compute inner loop \n'.format(num_rows-1, pe_y, int(compute_cycles_per_pe)))
					if retire_output==True and compute_cycles_per_pe>0:
						if batch_norm==True:
							f.write('BNORM {}_{} #Batch normalization \n'.format(num_rows-1, pe_y))
						if activation is not None:
							f.write('ACT {}_{} {} #Apply activation function \n'.format(num_rows-1, pe_y, activation))

				compute_cycles_per_pe = last_pe_load_x * last_pe_load_y * _b 

				f.write('ALU {}_{} {} #Compute inner loop \n'.format(num_rows-1, num_cols-1, int(compute_cycles_per_pe)))
				if retire_output==True and compute_cycles_per_pe>0:
					if batch_norm==True:
						f.write('BNORM {}_{} #Batch normalization \n'.format(num_rows-1, num_cols-1))
					if activation is not None:
						f.write('ACT {}_{} {} #Apply activation function \n'.format(num_rows-1, num_cols-1, activation))

				cycle += first_pe_load_x * first_pe_load_y * _b 
				if retire_output==True:
					cycle += 2 * first_pe_load_y * _b 
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


	dram_write += output_tile_size
	f.write('DRAM_WR {} #Write back output tile\n'.format(output_tile_size))
	cycle += int(math.ceil(float(output_tile_size) / DRAM_BW))

	f.write('EOL #End of layer\n')
	f.close()

	print('Total DRAM read simulated: {:,}'.format(dram_read))
	print('Total DRAM write simulated: {:,}'.format(dram_write))
	print('Total cycles simulated: {:,}'.format(cycle))


def add_residual_instr_writer(name, OH,OW,OC,B,oprec,N,M,activation,pooling=False, pool_window=None,):
	total_output_size = OH*OW*OC*B*oprec
	
	mem_chunk = N*M*oprec
	if pooling==True:
		mem_chunk = mem_chunk * pool_window[0] * pool_window[1]

	f= open("resnet50/"+name+".txt","w")

	mem_cnt = 0
	while mem_cnt < total_output_size:
		f.write('DRAM_RD {} #Read residual\n'.format(mem_chunk))
		f.write('DRAM_RD {} #Read output\n'.format(mem_chunk))
		
		for i in range(N):
			for j in range(M):
				f.write('ALU {}_{} {} #Add residual \n'.format(i, j, 1))
				f.write('ACT {}_{} {} #Apply activation function \n'.format(i, j, activation))
				if pooling==True:
					f.write('POOL {}_{} #Apply pooling function \n'.format(i, j))
		if pooling==True:
			f.write('DRAM_WR {} #Write back output\n'.format(int(math.ceil(mem_chunk/(pool_window[0]*pool_window[1])))))
		else:
			f.write('DRAM_WR {} #Write back output\n'.format(mem_chunk))
		mem_cnt += mem_chunk

	f.write('EOL #End of layer\n')



