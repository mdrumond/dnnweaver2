import logging
from dnnweaver2.tensorOps.cnn import *
from dnnweaver2.optimizer.optimizer import optimize_for_order, OP_TYPE

from compiler_helpers import conv2instr, lstm_arithm_instr, lstm_arithm_bp_instr, bnorm2instr, bnorm2instr_bp, maxpool2instr, maxpool2instr_bp, eltwise_instr
from custom_isa import ACTType, MACType

logger = logging.getLogger('{}.{}'.format(__name__, 'Compiler'))
logger.setLevel(logging.ERROR)

#In the original framework, all energy cost except dram is taken as zero. dram cost is taken as 6.e-3
energy_cost = (0,0,0,0,0,0,0,0,0,0,6.e-3)

lstm_activations = [ACTType.SIGMOID, ACTType.SIGMOID, ACTType.TANH, ACTType.SIGMOID]

def next_op(op_dict, opname):
	no_keys = len(op_dict.keys())
	next_ind = op_dict.keys().index(opname) + 1
	if next_ind < no_keys:
		return op_dict.keys()[next_ind]
	else:
		return None

def tensor_nodes(tensor_name, op_dict):
	src = []
	dst = []
	for opname in op_dict:
		op = op_dict[opname]
		if isinstance(op.output_tensors, list):
			for t in op.output_tensors:
				if tensor_name is t.name:
					src.append(op)			
		else:
			if tensor_name is op.output_tensors.name:
				src.append(op)
		for t in op.input_tensors:
			if tensor_name is t.name:
				dst.append(op)
	return (src,dst)

def get_src_error(tensor, op_dict):
	(src,dst) = tensor_nodes(tensor.name, op_dict)
	logger.debug('Tensor name: {} src:{} dst:{}'.format(tensor.name, src, dst))

	if len(src)==0:
		return None

	if isinstance(src[0].output_tensors, list):
		no_src_out_tensors = len(src[0].output_tensors)
		for t in range(no_src_out_tensors):
			if src[0].output_tensors[t].name is tensor.name:
				out_ind = t
		return src[0].output_errors[out_ind]
	else:
		return src[0].output_errors


def compile_lstm(dir_name, graph, acc_obj):
	f = open(dir_name+"lstm.bin", "wb")
	f.close()

	op_dict = graph.op_registry

	instr = []	

	prec = acc_obj.prec

	f_tensor = open(dir_name+"tensors.txt", "w")
	tensor_id = {}
	tid = 0
	for i in graph.tensor_registry:
		(src,dst) = tensor_nodes(i, op_dict)
		
		logger.debug('tensor_name:{} \t\t src:{} -> dst:{}'.format(i,src,dst))

		f_tensor.write('{} {} {}\n'.format(tid, i, graph.tensor_registry[i].shape))
		tensor_id[i] = tid
		tid += 1
	f_tensor.close()

	for opname in op_dict:
		op = op_dict[opname]
		print('Op name: {}'.format(opname))
		filename = opname.replace("/","_")

		if isinstance(op, LSTM_FC):
			data = op.input_tensors[0]
			wf = op.input_tensors[1]
			wi = op.input_tensors[2]
			wc = op.input_tensors[3]
			wo = op.input_tensors[4]
			bf = op.input_tensors[5]
			bi = op.input_tensors[6]
			bc = op.input_tensors[7]
			bo = op.input_tensors[8]

			out_f = op.output_tensors[0]
			out_i = op.output_tensors[1]
			out_c = op.output_tensors[2]
			out_o = op.output_tensors[3]

			tensor_ids = []
			tensor_ids.append(tensor_id[data.name])
			tensor_ids.append(tensor_id[wf.name])
			tensor_ids.append(tensor_id[wi.name])
			tensor_ids.append(tensor_id[wc.name])
			tensor_ids.append(tensor_id[wo.name])
			tensor_ids.append(tensor_id[bf.name])			
			tensor_ids.append(tensor_id[bi.name])	
			tensor_ids.append(tensor_id[bc.name])	
			tensor_ids.append(tensor_id[bo.name])	
			tensor_ids.append(tensor_id[out_f.name])			
			tensor_ids.append(tensor_id[out_i.name])	
			tensor_ids.append(tensor_id[out_c.name])	
			tensor_ids.append(tensor_id[out_o.name])	

			tensor_ids = tuple(tensor_ids)
			logger.debug('tensor_ids:{}'.format(tensor_ids))

			K = wf.fpga_shape[-2]
			O = out_f.fpga_shape[-2]
			S = op.stride[-2]
			IC = wf.fpga_shape[-1]
			OC = wf.fpga_shape[-4]
			B = data.fpga_shape[-4]
			G = 4 #There are 4 gates in an LSTM cell (4 parallel FC layers)
			logger.debug('K:{} O:{} S:{} IC:{} OC:{} B:{} G:{}'.format(K,O,S,IC,OC,B,G))

			conv_params = (acc_obj, K, O, S, IC, OC, B, G, energy_cost)

			_, _, stats = optimize_for_order(conv_params, sequential=False, op_type=OP_TYPE.LSTM_FW)
			instr = conv2instr(instr, stats, dir_name, filename, tensor_ids, conv_params, op_type=OP_TYPE.LSTM_FW, mac_type=MACType.FC, activation=lstm_activations)

		elif isinstance(op, LSTM_ARITH):
			Cin = op.input_tensors[0]
			of = op.input_tensors[1]
			oi = op.input_tensors[2]
			oc = op.input_tensors[3]
			oo = op.input_tensors[4]

			Cout = op.output_tensors[0]
			hout = op.output_tensors[1]

			tensor_ids = []
			tensor_ids.append(tensor_id[Cin.name])
			tensor_ids.append(tensor_id[of.name])
			tensor_ids.append(tensor_id[oi.name])
			tensor_ids.append(tensor_id[oc.name])
			tensor_ids.append(tensor_id[oo.name])
			tensor_ids.append(tensor_id[Cout.name])			
			tensor_ids.append(tensor_id[hout.name])	

			tensor_ids = tuple(tensor_ids)
			logger.debug('tensor_ids:{}'.format(tensor_ids))
			
			H = Cin.shape[-1]
			lstm_arithm_instr(instr, dir_name, filename, tensor_ids, H, acc_obj)

		elif isinstance(op, Fork):
			tensor_id[op.output_tensors[0].name] = tensor_id[op.input_tensors[0].name]
			tensor_id[op.output_tensors[1].name] = tensor_id[op.input_tensors[0].name]

		elif isinstance(op, FC):
			data = op.input_tensors[0]
			weight = op.input_tensors[1]
			bias = op.input_tensors[2]
			out = op.output_tensors

			tensor_ids = (tensor_id[data.name], tensor_id[weight.name], tensor_id[bias.name], tensor_id[out.name])
			logger.debug('tensor_ids:{}'.format(tensor_ids))

			logger.debug('Input tensor: {}'.format(data.shape))
			logger.debug('Weight tensor: {}'.format(weight.shape))
			logger.debug('Bias tensor: {}'.format(bias.shape))
			logger.debug('Output tensor: {}'.format(out.shape))

			K = weight.fpga_shape[-2]
			O = out.fpga_shape[-2]
			S = op.stride[-2]
			IC = weight.fpga_shape[-1]
			OC = weight.fpga_shape[-4]
			B = data.fpga_shape[-4]
			G = 1 #There are 4 gates in an LSTM cell (4 parallel FC layers), 1 is equal to conventional layers
			logger.debug('K:{} O:{} S:{} IC:{} OC:{} B:{} G:{}'.format(K,O,S,IC,OC,B,G))

			conv_params = (acc_obj, K, O, S, IC, OC, B, G, energy_cost)

			_, _, stats = optimize_for_order(conv_params, sequential=False, op_type=OP_TYPE.FW)
			instr = conv2instr(instr, stats, dir_name, filename, tensor_ids, conv_params, op_type=OP_TYPE.FW, mac_type=MACType.FC, activation=None)

		elif isinstance(op, CellState):
			tensor_id[op.output_tensors.name] = tensor_id[op.input_tensors[0].name]

		else:
			assert False, 'Graph node not implemented, revisit the code'

		f = open(dir_name+"lstm.bin", "ab")
		instr_bin = bytearray(instr)
		f.write(instr_bin)
		f.close()
		instr = []


def compile_lstm_bp(dir_name, graph, acc_obj):
	f = open(dir_name+"lstm_bp.bin", "wb")
	f.close()

	op_dict = graph.op_registry

	instr = []	

	prec = acc_obj.prec

	f_tensor = open(dir_name+"tensors.txt", "w")
	tensor_id = {}
	tid = 0
	for i in graph.tensor_registry:
		(src,dst) = tensor_nodes(i, op_dict)
		
		logger.debug('tensor_name:{} \t\t src:{} -> dst:{}'.format(i,src,dst))
		
		f_tensor.write('{} {} {}\n'.format(tid, i, graph.tensor_registry[i].shape))
		tensor_id[i] = tid
		tid += 1
	f_tensor.close()

	for opname in reversed(op_dict):
		op = op_dict[opname]
		print('Op name: {}'.format(opname))
		filename = opname.replace("/","_")
	
		if isinstance(op, FC):
			data = op.input_tensors[0]
			weight = op.input_tensors[1]
			out = op.output_tensors
			out_error = op.output_errors
			w_grad = op.weight_grads

			logger.debug('Input tensor: {}'.format(data.shape))
			logger.debug('Weight tensor: {}'.format(weight.shape))
			logger.debug('Output tensor: {}'.format(out.shape))			
			logger.debug('out_error tensor: {}'.format(out_error.shape))
			logger.debug('w_grad tensor: {}'.format(w_grad.shape))

			K = weight.fpga_shape[-2]
			O = out.fpga_shape[-2]
			S = op.stride[-2]
			IC = weight.fpga_shape[-1]
			OC = weight.fpga_shape[-4]
			B = data.fpga_shape[-4]
			G = 1 #There are 4 gates in an LSTM cell (4 parallel FC layers), 1 is equal to conventional layers
			logger.debug('K:{} O:{} S:{} IC:{} OC:{} B:{} G:{}'.format(K,O,S,IC,OC,B,G))

			conv_params = (acc_obj, K, O, S, IC, OC, B, G, energy_cost)

			#error -> grads
			tensor_ids = (tensor_id[data.name], tensor_id[w_grad.name], None, tensor_id[out_error.name])
			_, _, stats = optimize_for_order(conv_params, sequential=False, op_type=OP_TYPE.GD)
			instr = conv2instr(instr, stats, dir_name, filename+'_gd', tensor_ids, conv_params, op_type=OP_TYPE.GD, mac_type=MACType.FC, activation=None)

			#error -> prev. error
			prev_error = get_src_error(data, op_dict)
			tensor_ids = (tensor_id[prev_error.name], tensor_id[weight.name], None, tensor_id[out_error.name])
			_, _, stats = optimize_for_order(conv_params, sequential=False, op_type=OP_TYPE.BP)
			instr = conv2instr(instr, stats, dir_name, filename+'_bp', tensor_ids, conv_params, op_type=OP_TYPE.BP, mac_type=MACType.FC, activation=None)

		elif isinstance(op, Fork):
			O = op.output_tensors[0].fpga_shape[-2]
			OC = op.output_tensors[0].fpga_shape[-1]
			B = op.output_tensors[0].fpga_shape[-4]

			prev_error = get_src_error(op.input_tensors[0], op_dict)
			tensor_ids = (tensor_id[op.output_errors[0].name], tensor_id[op.output_errors[1].name], tensor_id[prev_error.name])
			print('prev_error:{}'.format(prev_error.name))

			#Add errors due to residual branch
			instr = eltwise_instr(instr, dir_name, filename, tensor_ids, O, O, OC, B, acc_obj, vector_op='Add', activation=None)
		
		elif isinstance(op, LSTM_ARITH):
			Cin, of, oi, oc, oo = op.input_tensors
			Cout, hout = op.output_tensors
			Cout_bp, hout_bp = op.output_errors

			Cin_bp = get_src_error(op.input_tensors[0], op_dict)			
			of_bp = get_src_error(op.input_tensors[1], op_dict)
			oi_bp = get_src_error(op.input_tensors[2], op_dict)
			oc_bp = get_src_error(op.input_tensors[3], op_dict)
			oo_bp = get_src_error(op.input_tensors[4], op_dict)

			tensor_ids = []
			tensor_ids.append(tensor_id[Cin.name])
			tensor_ids.append(tensor_id[Cin_bp.name])
			tensor_ids.append(tensor_id[of.name])
			tensor_ids.append(tensor_id[of_bp.name])
			tensor_ids.append(tensor_id[oi.name])
			tensor_ids.append(tensor_id[oi_bp.name])
			tensor_ids.append(tensor_id[oc.name])
			tensor_ids.append(tensor_id[oc_bp.name])
			tensor_ids.append(tensor_id[oo.name])
			tensor_ids.append(tensor_id[oo_bp.name])
			tensor_ids.append(tensor_id[Cout.name])
			tensor_ids.append(tensor_id[Cout_bp.name])
			tensor_ids.append(tensor_id[hout.name])
			tensor_ids.append(tensor_id[hout_bp.name])

			H = Cin.shape[-1]

			instr = lstm_arithm_bp_instr(instr, dir_name, filename, tensor_ids, H, acc_obj)


		elif isinstance(op, LSTM_FC):
			data = op.input_tensors[0]
			wf = op.input_tensors[1]
			wi = op.input_tensors[2]
			wc = op.input_tensors[3]
			wo = op.input_tensors[4]
			bf = op.input_tensors[5]
			bi = op.input_tensors[6]
			bc = op.input_tensors[7]
			bo = op.input_tensors[8]

			out_f = op.output_tensors[0]
			out_i = op.output_tensors[1]
			out_c = op.output_tensors[2]
			out_o = op.output_tensors[3]


			data = op.input_tensors[0]
			wf = op.input_tensors[1]
			wi = op.input_tensors[2]
			wc = op.input_tensors[3]
			wo = op.input_tensors[4]
			wf_grad = op.weight_grads[0]
			wi_grad = op.weight_grads[1]
			wc_grad = op.weight_grads[2]
			wo_grad = op.weight_grads[3]

			bf = op.input_tensors[5]
			bi = op.input_tensors[6]
			bc = op.input_tensors[7]
			bo = op.input_tensors[8]

			out_f = op.output_tensors[0]
			out_i = op.output_tensors[1]
			out_c = op.output_tensors[2]
			out_o = op.output_tensors[3]
			out_f_bp = op.output_errors[0]
			out_i_bp = op.output_errors[1]
			out_c_bp = op.output_errors[2]
			out_o_bp = op.output_errors[3]

			K = wf.fpga_shape[-2]
			O = out_f.fpga_shape[-2]
			S = op.stride[-2]
			IC = wf.fpga_shape[-1]
			OC = wf.fpga_shape[-4]
			B = data.fpga_shape[-4]
			G = 4 #There are 4 gates in an LSTM cell (4 parallel FC layers)
			logger.debug('K:{} O:{} S:{} IC:{} OC:{} B:{} G:{}'.format(K,O,S,IC,OC,B,G))

			conv_params = (acc_obj, K, O, S, IC, OC, B, G, energy_cost)

			tensor_ids = []
			tensor_ids.append(tensor_id[data.name])
			tensor_ids.append(tensor_id[wf_grad.name])
			tensor_ids.append(tensor_id[wi_grad.name])
			tensor_ids.append(tensor_id[wc_grad.name])
			tensor_ids.append(tensor_id[wo_grad.name])
			tensor_ids.append(None)			
			tensor_ids.append(None)	
			tensor_ids.append(None)	
			tensor_ids.append(None)	
			tensor_ids.append(tensor_id[out_f_bp.name])			
			tensor_ids.append(tensor_id[out_i_bp.name])	
			tensor_ids.append(tensor_id[out_c_bp.name])	
			tensor_ids.append(tensor_id[out_o_bp.name])	

			tensor_ids = tuple(tensor_ids)
			logger.debug('tensor_ids:{}'.format(tensor_ids))

			#error -> grads
			_, _, stats = optimize_for_order(conv_params, sequential=False, op_type=OP_TYPE.LSTM_GD)
			instr = conv2instr(instr, stats, dir_name, filename+'_gd', tensor_ids, conv_params, op_type=OP_TYPE.LSTM_GD, mac_type=MACType.FC, activation=None)


			#error -> prev. error
			prev_error = get_src_error(data, op_dict)


			tensor_ids = []
			tensor_ids.append(tensor_id[prev_error.name])
			tensor_ids.append(tensor_id[wf.name])
			tensor_ids.append(tensor_id[wi.name])
			tensor_ids.append(tensor_id[wc.name])
			tensor_ids.append(tensor_id[wo.name])
			tensor_ids.append(None)	
			tensor_ids.append(None)	
			tensor_ids.append(None)	
			tensor_ids.append(None)	
			tensor_ids.append(tensor_id[out_f_bp.name])			
			tensor_ids.append(tensor_id[out_i_bp.name])	
			tensor_ids.append(tensor_id[out_c_bp.name])	
			tensor_ids.append(tensor_id[out_o_bp.name])	

			tensor_ids = tuple(tensor_ids)
			logger.debug('tensor_ids:{}'.format(tensor_ids))

			_, _, stats = optimize_for_order(conv_params, sequential=False, op_type=OP_TYPE.LSTM_BP)
			instr = conv2instr(instr, stats, dir_name, filename+'_bp', tensor_ids, conv_params, op_type=OP_TYPE.LSTM_BP, mac_type=MACType.FC, activation=None)

		elif isinstance(op, CellState):
			tensor_id[op.output_tensors.name] = tensor_id[op.input_tensors[0].name]

		else:
			assert False


def compile_graph(dir_name, graph, acc_obj):
	logger.debug('Opening binary file in {}'.format(dir_name))
	
	f = open(dir_name+"resnet50.bin", "wb")
	f.close()

	instr = []	

	prec = acc_obj.prec

	op_dict = graph.op_registry

	f_tensor = open(dir_name+"tensors.txt", "w")
	tensor_id = {}
	tid = 0
	for i in graph.tensor_registry:
		(src,dst) = tensor_nodes(i, op_dict)
		
		logger.debug('tensor_name:{} \t\t src:{} -> dst:{}'.format(i,src,dst))

		f_tensor.write('{} {} {}\n'.format(tid, i, graph.tensor_registry[i].shape))
		tensor_id[i] = tid
		tid += 1
	f_tensor.close()

	

	for opname in op_dict:
		op = op_dict[opname]
		print('Op name: {}'.format(opname))
		filename = opname.replace("/","_")

		if isinstance(op, Convolution):
			data = op.input_tensors[0]
			weight = op.input_tensors[1]
			bias = op.input_tensors[2]
			out = op.output_tensors

			tensor_ids = (tensor_id[data.name], tensor_id[weight.name], tensor_id[bias.name], tensor_id[out.name])
			logger.debug('tensor_ids:{}'.format(tensor_ids))

			logger.debug('Input tensor: {} {}'.format(data.name, data.shape))
			logger.debug('Weight tensor: {} {}'.format(weight.name, weight.shape))
			logger.debug('Bias tensor: {} {}'.format(bias.name, bias.shape))
			logger.debug('Output tensor: {} {}'.format(out.name, out.shape))

			K = weight.fpga_shape[-2]
			O = out.fpga_shape[-2]
			S = op.stride[-2]
			IC = weight.fpga_shape[-1]
			OC = weight.fpga_shape[-4]
			B = data.fpga_shape[-4]
			G = 1 #There are 4 gates in an LSTM cell (4 parallel FC layers), 1 is equal to conventional layers
			logger.debug('K:{} O:{} S:{} IC:{} OC:{} B:{} G:{}'.format(K,O,S,IC,OC,B,G))

			conv_params = (acc_obj, K, O, S, IC, OC, B, G, energy_cost)

			_, _, stats = optimize_for_order(conv_params, sequential=False, op_type=OP_TYPE.FW)
			instr = conv2instr(instr, stats, dir_name, filename, tensor_ids, conv_params, op_type=OP_TYPE.FW, mac_type=MACType.CONV, activation=None)

		elif isinstance(op, BNorm):
			tensor_ids = (tensor_id[op.input_tensors[0].name], tensor_id[op.input_tensors[1].name], tensor_id[op.input_tensors[2].name], tensor_id[op.output_tensors.name])
			logger.debug('tensor_ids:{}'.format(tensor_ids))

			O = op.output_tensors.fpga_shape[-2]
			OC = op.output_tensors.fpga_shape[-1]
			B = op.output_tensors.fpga_shape[-4]

			next = next_op(op_dict, opname)
			if isinstance(op_dict[next], ReLU) == True:
				activation=ACTType.RELU
			else:
				activation=None

			logger.debug('activation:{}'.format(activation))

			instr = bnorm2instr(instr, dir_name, filename, tensor_ids, O, O, OC, B, acc_obj, activation)

		elif isinstance(op, Add):
			tensor_ids = (tensor_id[op.input_tensors[0].name], tensor_id[op.input_tensors[1].name], tensor_id[op.output_tensors.name])
			logger.debug('tensor_ids:{}'.format(tensor_ids))

			O = op.output_tensors.fpga_shape[-2]
			OC = op.output_tensors.fpga_shape[-1]
			B = op.output_tensors.fpga_shape[-4]

			next = next_op(op_dict, opname)
			if isinstance(op_dict[next], ReLU) == True:
				activation=ACTType.RELU
			else:
				activation=None
			logger.debug('activation:{}'.format(activation))

			instr = eltwise_instr(instr, dir_name, filename, tensor_ids, O, O, OC, B, acc_obj, vector_op='Add', activation=activation)

		elif isinstance(op, Fork):
			tensor_id[op.output_tensors[0].name] = tensor_id[op.input_tensors[0].name]
			tensor_id[op.output_tensors[1].name] = tensor_id[op.input_tensors[0].name]

		elif isinstance(op, MaxPooling):
			tensor_ids = (tensor_id[op.input_tensors[0].name], tensor_id[op.output_tensors.name])
			logger.debug('tensor_ids:{}'.format(tensor_ids))

			O = op.output_tensors.fpga_shape[-2]
			OC = op.output_tensors.fpga_shape[-1]
			B = op.output_tensors.fpga_shape[-4]	
			logger.debug('O:{} OC:{} B:{}'.format(O,OC,B))

			instr = maxpool2instr(instr, dir_name, filename, tensor_ids, O, O, OC, B, acc_obj, pool_kernel=op.pooling_kernel)		

		elif isinstance(op, FC):
			data = op.input_tensors[0]
			weight = op.input_tensors[1]
			bias = op.input_tensors[2]
			out = op.output_tensors

			tensor_ids = (tensor_id[data.name], tensor_id[weight.name], tensor_id[bias.name], tensor_id[out.name])
			logger.debug('tensor_ids:{}'.format(tensor_ids))

			logger.debug('Input tensor: {}'.format(data.shape))
			logger.debug('Weight tensor: {}'.format(weight.shape))
			logger.debug('Bias tensor: {}'.format(bias.shape))
			logger.debug('Output tensor: {}'.format(out.shape))

			K = weight.fpga_shape[-2]
			O = out.fpga_shape[-2]
			S = op.stride[-2]
			IC = weight.fpga_shape[-1]
			OC = weight.fpga_shape[-4]
			B = data.fpga_shape[-4]
			G = 1 #There are 4 gates in an LSTM cell (4 parallel FC layers), 1 is equal to conventional layers
			logger.debug('K:{} O:{} S:{} IC:{} OC:{} B:{} G:{}'.format(K,O,S,IC,OC,B,G))

			conv_params = (acc_obj, K, O, S, IC, OC, B, G, energy_cost)

			_, _, stats = optimize_for_order(conv_params, sequential=False, op_type=OP_TYPE.FW)
			instr = conv2instr(instr, stats, dir_name, filename, tensor_ids, conv_params, op_type=OP_TYPE.FW, mac_type=MACType.FC, activation=None)
		else:
			print('Ineffective layer:{} Connecting input to output:'.format(opname))
			tensor_id[op.output_tensors.name] = tensor_id[op.input_tensors[0].name]
			print('tensor_id:{}'.format(tensor_id[op.input_tensors[0].name]))
			print('tensor_id:{}'.format(tensor_id[op.output_tensors.name]))

		f = open(dir_name+"resnet50.bin", "ab")
		instr_bin = bytearray(instr)
		f.write(instr_bin)
		f.close()
		instr = []

# https://www.jefkine.com/general/2016/09/05/backpropagation-in-convolutional-neural-networks/
# http://neuralnetworksanddeeplearning.com/chap2.html
def compile_graph_bp(dir_name, graph, acc_obj):
	f = open(dir_name+"resnet50.bin", "wb")
	f.close()

	instr = []	

	prec = acc_obj.prec

	op_dict = graph.op_registry

	f_tensor = open(dir_name+"tensors.txt", "w")
	tensor_id = {}
	tid = 0
	for i in graph.tensor_registry:
		(src,dst) = tensor_nodes(i, op_dict)
		
		logger.debug('tensor_name:{} \t\t src:{} -> dst:{}'.format(i,src,dst))

		f_tensor.write('{} {} {}\n'.format(tid, i, graph.tensor_registry[i].shape))
		tensor_id[i] = tid
		tid += 1
	f_tensor.close()

	for opname in reversed(op_dict):
		op = op_dict[opname]
		print('Op name: {}_bp'.format(opname))
		filename = opname.replace("/","_")

		if isinstance(op, FC):
			data = op.input_tensors[0]
			weight = op.input_tensors[1]
			out = op.output_tensors
			out_error = op.output_errors
			w_grad = op.weight_grads

			logger.debug('Input tensor: {}'.format(data.shape))
			logger.debug('Weight tensor: {}'.format(weight.shape))
			logger.debug('Output tensor: {}'.format(out.shape))			
			logger.debug('out_error tensor: {}'.format(out_error.shape))
			logger.debug('w_grad tensor: {}'.format(w_grad.shape))

			K = weight.fpga_shape[-2]
			O = out.fpga_shape[-2]
			S = op.stride[-2]
			IC = weight.fpga_shape[-1]
			OC = weight.fpga_shape[-4]
			B = data.fpga_shape[-4]
			G = 1 #There are 4 gates in an LSTM cell (4 parallel FC layers), 1 is equal to conventional layers
			logger.debug('K:{} O:{} S:{} IC:{} OC:{} B:{} G:{}'.format(K,O,S,IC,OC,B,G))

			conv_params = (acc_obj, K, O, S, IC, OC, B, G, energy_cost)

			#error -> grads
			tensor_ids = (tensor_id[data.name], tensor_id[w_grad.name], None, tensor_id[out_error.name])
			_, _, stats = optimize_for_order(conv_params, sequential=False, op_type=OP_TYPE.GD)
			instr = conv2instr(instr, stats, dir_name, filename+'_gd', tensor_ids, conv_params, op_type=OP_TYPE.GD, mac_type=MACType.FC, activation=None)

			#error -> prev. error
			prev_error = get_src_error(data, op_dict)
			tensor_ids = (tensor_id[prev_error.name], tensor_id[weight.name], None, tensor_id[out_error.name])
			_, _, stats = optimize_for_order(conv_params, sequential=False, op_type=OP_TYPE.BP)
			instr = conv2instr(instr, stats, dir_name, filename+'_bp', tensor_ids, conv_params, op_type=OP_TYPE.BP, mac_type=MACType.FC, activation=None)

		elif isinstance(op, MaxPooling):
			O = op.output_tensors.fpga_shape[-2]
			OC = op.output_tensors.fpga_shape[-1]
			B = op.output_tensors.fpga_shape[-4]	
			logger.debug('O:{} OC:{} B:{}'.format(O,OC,B))

			input_tensor = op.input_tensors[0]
			prev_error = get_src_error(input_tensor, op_dict)
			tensor_ids = (tensor_id[prev_error.name], tensor_id[op.output_errors.name])

			instr = maxpool2instr_bp(instr, dir_name, filename, tensor_ids, O, O, OC, B, acc_obj, pool_kernel=op.pooling_kernel)	

		elif isinstance(op, ReLU):
			O = op.output_tensors.fpga_shape[-2]
			OC = op.output_tensors.fpga_shape[-1]
			B = op.output_tensors.fpga_shape[-4]	

			input_tensor = op.input_tensors[0]
			prev_error = get_src_error(input_tensor, op_dict)
			tensor_ids = (tensor_id[op.output_errors.name], tensor_id[input_tensor.name], tensor_id[prev_error.name])

			#multiply error with act. derivation
			instr = eltwise_instr(instr, dir_name, filename, tensor_ids, O, O, OC, B, acc_obj, vector_op='Mult_der', activation=None)

		elif isinstance(op, BNorm):
			O = op.output_tensors.fpga_shape[-2]
			OC = op.output_tensors.fpga_shape[-1]
			B = op.output_tensors.fpga_shape[-4]

			#h_id, dh_id, gamma_id, dgamma_id, beta_id, dbeta_id, dy_id = tensor_ids
			input_tensor = op.input_tensors[0]
			gamma_tensor = op.input_tensors[1]
			beta_tensor = op.input_tensors[2]
			dgamma_tensor = op.gamma_grads
			dbeta_tensor = op.beta_grads			

			prev_error = get_src_error(input_tensor, op_dict)

			tensor_ids = []
			tensor_ids.append(tensor_id[input_tensor.name])
			tensor_ids.append(tensor_id[prev_error.name])
			tensor_ids.append(tensor_id[gamma_tensor.name])
			tensor_ids.append(tensor_id[dgamma_tensor.name])
			tensor_ids.append(tensor_id[beta_tensor.name])
			tensor_ids.append(tensor_id[dbeta_tensor.name])
			tensor_ids.append(tensor_id[op.output_errors.name])

			instr = bnorm2instr_bp(instr, dir_name, filename, tensor_ids, O, O, OC, B, acc_obj)

		elif isinstance(op, Convolution):
			data = op.input_tensors[0]
			weight = op.input_tensors[1]
			bias = op.input_tensors[2]
			out = op.output_tensors
			out_error = op.output_errors
			w_grad = op.weight_grads

			logger.debug('Input tensor: {}'.format(data.shape))
			logger.debug('Weight tensor: {}'.format(weight.shape))
			logger.debug('Bias tensor: {}'.format(bias.shape))
			logger.debug('Output tensor: {}'.format(out.shape))			
			logger.debug('out_error tensor: {}'.format(out_error.shape))
			logger.debug('w_grad tensor: {}'.format(w_grad.shape))	

			K = weight.fpga_shape[-2]
			O = out.fpga_shape[-2]
			S = op.stride[-2]
			IC = weight.fpga_shape[-1]
			OC = weight.fpga_shape[-4]
			B = data.fpga_shape[-4]
			G = 1 #There are 4 gates in an LSTM cell (4 parallel FC layers), 1 is equal to conventional layers
			logger.debug('K:{} O:{} S:{} IC:{} OC:{} B:{} G:{}'.format(K,O,S,IC,OC,B,G))

			conv_params = (acc_obj, K, O, S, IC, OC, B, G, energy_cost)


			#error -> grads
			tensor_ids = (tensor_id[data.name], tensor_id[w_grad.name], None, tensor_id[out_error.name])
			_, _, stats = optimize_for_order(conv_params, sequential=False, op_type=OP_TYPE.GD)
			instr = conv2instr(instr, stats, dir_name, filename+'_gd', tensor_ids, conv_params, op_type=OP_TYPE.GD, mac_type=MACType.CONV, activation=None)


			#error -> prev. error
			prev_error = get_src_error(data, op_dict)
			if prev_error==None:
				break #We reached the input layer

			tensor_ids = (tensor_id[prev_error.name], tensor_id[weight.name], None, tensor_id[out_error.name])
			_, _, stats = optimize_for_order(conv_params, sequential=False, op_type=OP_TYPE.BP)
			instr = conv2instr(instr, stats, dir_name, filename+'_bp', tensor_ids, conv_params, op_type=OP_TYPE.BP, mac_type=MACType.CONV, activation=None)

		elif isinstance(op, Add):
			prev_error1 = get_src_error(op.input_tensors[0], op_dict)
			prev_error2 = get_src_error(op.input_tensors[1], op_dict)

			tensor_id[prev_error1.name] = tensor_id[op.output_errors.name]
			tensor_id[prev_error2.name] = tensor_id[op.output_errors.name]

		elif isinstance(op, Fork):
			O = op.output_tensors[0].fpga_shape[-2]
			OC = op.output_tensors[0].fpga_shape[-1]
			B = op.output_tensors[0].fpga_shape[-4]

			prev_error = get_src_error(op.input_tensors[0], op_dict)
			tensor_ids = (tensor_id[op.output_errors[0].name], tensor_id[op.output_errors[1].name], tensor_id[prev_error.name])


			#Add errors due to residual branch
			instr = eltwise_instr(instr, dir_name, filename, tensor_ids, O, O, OC, B, acc_obj, vector_op='Add', activation=None)

		else:
			print('Ineffective layer:{} Connecting input to output:'.format(opname))
			prev_error = get_src_error(op.input_tensors[0], op_dict)
			tensor_id[prev_error.name] = tensor_id[op.output_errors.name]


		f = open(dir_name+"resnet50.bin", "ab")
		instr_bin = bytearray(instr)
		f.write(instr_bin)
		f.close()
		instr = []









