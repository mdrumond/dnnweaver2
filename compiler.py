import logging
from dnnweaver2.tensorOps.cnn import *
from dnnweaver2.optimizer.optimizer import optimize_for_order, optimize_for_order_bp

from compiler_helpers import conv2instr, conv2instr_bp, bnorm2instr, bnorm2instr_bp, maxpool2instr, maxpool2instr_bp, eltwise_instr
from custom_isa import ACTType, MACType

logger = logging.getLogger('{}.{}'.format(__name__, 'Compiler'))
logger.setLevel(logging.DEBUG)


def next_op(op_dict, opname):
	no_keys = len(op_dict.keys())
	next_ind = op_dict.keys().index(opname) + 1
	if next_ind < no_keys:
		return op_dict.keys()[next_ind]
	else:
		return None

def compile_graph(dir_name, graph, acc_obj):
	logger.debug('Opening binary file in {}'.format(dir_name))
	f = open(dir_name+"resnet50.bin", "wb")

	instr = []	

	prec = acc_obj.prec

	tensor_id = {}
	tid = 0
	for i in graph.tensor_registry:
		logger.debug('tensor_name:{}'.format(i))
		tensor_id[i] = tid
		tid += 1

	op_dict = graph.op_registry

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

			logger.debug('Input tensor: {}'.format(data.shape))
			logger.debug('Weight tensor: {}'.format(weight.shape))
			logger.debug('Bias tensor: {}'.format(bias.shape))
			logger.debug('Output tensor: {}'.format(out.shape))

			K = weight.fpga_shape[-2]
			O = out.fpga_shape[-2]
			S = op.stride[-1]
			IC = weight.fpga_shape[-1]
			OC = weight.fpga_shape[-4]
			B = data.fpga_shape[-4]
			logger.debug('K:{} O:{} S:{} IC:{} OC:{} B:{}'.format(K,O,S,IC,OC,B))

			#energy_cost and im2col are hardcoded
			energy_cost = (0,0,0,0,0,0,0,0,0,0)
			im2col = False
			conv_params = (acc_obj, K, O, S, IC, OC, B, prec, im2col, energy_cost)

			_, _, stats = optimize_for_order(conv_params, sequential=False, pool_kernel=None, pool_stride=None)
			instr = conv2instr(instr, stats, dir_name, filename, tensor_ids, O, O, OC, IC, B, K, S, acc_obj, mac_type=MACType.CONV, activation=None, transpose_weight=False)

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
			S = op.stride[-1]
			IC = weight.fpga_shape[-1]
			OC = weight.fpga_shape[-4]
			B = data.fpga_shape[-4]
			logger.debug('K:{} O:{} S:{} IC:{} OC:{} B:{}'.format(K,O,S,IC,OC,B))

			#energy_cost and im2col are hardcoded
			energy_cost = (0,0,0,0,0,0,0,0,0,0)
			im2col = False
			conv_params = (acc_obj, K, O, S, IC, OC, B, prec, im2col, energy_cost)

			_, _, stats = optimize_for_order(conv_params, sequential=False, pool_kernel=None, pool_stride=None)
			instr = conv2instr(instr, stats, dir_name, filename, tensor_ids, O, O, OC, IC, B, K, S, acc_obj, mac_type=MACType.FC, activation=None, transpose_weight=False)

		else:
			print('Unoptimized layer:{} Connecting input to output:'.format(opname))
			tensor_id[op.output_tensors.name] = tensor_id[op.input_tensors[0].name]
			print('tensor_id:{}'.format(tensor_id[op.input_tensors[0].name]))
			print('tensor_id:{}'.format(tensor_id[op.output_tensors.name]))


	instr_bin = bytearray(instr)
	f.write(instr_bin)
	f.close()


def compile_graph_bp(dir_name, graph, acc_obj):
	f = open(dir_name+"resnet50.bin", "wb")

	instr = []	

	prec = acc_obj.prec

	tensor_id = {}
	tid = 0
	for i in graph.tensor_registry:
		logger.debug('tensor_name:{}'.format(i))
		tensor_id[i] = tid
		tid += 1

	op_dict = graph.op_registry

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
			S = op.stride[-1]
			IC = weight.fpga_shape[-1]
			OC = weight.fpga_shape[-4]
			B = data.fpga_shape[-4]
			logger.debug('K:{} O:{} S:{} IC:{} OC:{} B:{}'.format(K,O,S,IC,OC,B))

			#energy_cost and im2col are hardcoded
			energy_cost = (0,0,0,0,0,0,0,0,0,0)
			im2col = False
			conv_params = (acc_obj, K, O, S, IC, OC, B, prec, im2col, energy_cost)

			#error -> grads
			tensor_ids = (tensor_id[out_error.name], tensor_id[data.name], None, tensor_id[w_grad.name])
			_, _, stats = optimize_for_order_bp(conv_params, sequential=False, pool_kernel=None, pool_stride=None)
			instr = conv2instr_bp(instr, stats, dir_name, filename+'_grad', tensor_ids, O, O, OC, IC, B, K, S, acc_obj, activation=None)

			#error -> prev. error
			tensor_ids = (tensor_id[out_error.name], tensor_id[weight.name], None, tensor_id[out.name])
			_, _, stats = optimize_for_order(conv_params, sequential=False, pool_kernel=None, pool_stride=None)
			instr = conv2instr(instr, stats, dir_name, filename+'_bp', tensor_ids, O, O, OC, IC, B, K, S, acc_obj, None)

		if isinstance(op, MaxPooling):
			O = op.output_tensors.fpga_shape[-2]
			OC = op.output_tensors.fpga_shape[-1]
			B = op.output_tensors.fpga_shape[-4]	
			logger.debug('O:{} OC:{} B:{}'.format(O,OC,B))

			tensor_ids = (tensor_id[op.input_tensors[0].name], tensor_id[op.output_tensors.name])

			instr = maxpool2instr_bp(instr, dir_name, filename, tensor_ids, O, O, OC, B, acc_obj, pool_kernel=op.pooling_kernel)	

		if isinstance(op, ReLU):
			O = op.output_tensors.fpga_shape[-2]
			OC = op.output_tensors.fpga_shape[-1]
			B = op.output_tensors.fpga_shape[-4]	

			tensor_ids = (tensor_id[op.input_tensors[0].name], tensor_id[op.input_tensors[0].name], tensor_id[op.output_tensors.name])

			#multiply error with act. derivation
			instr = eltwise_instr(instr, dir_name, filename, tensor_ids, O, O, OC, B, acc_obj, vector_op='Mult', activation=ACTType.RELU)

		if isinstance(op, BNorm):
			O = op.output_tensors.fpga_shape[-2]
			OC = op.output_tensors.fpga_shape[-1]
			B = op.output_tensors.fpga_shape[-4]

			tensor_ids = (tensor_id[op.input_tensors[0].name], tensor_id[op.input_tensors[1].name], tensor_id[op.input_tensors[2].name], tensor_id[op.output_tensors.name])

			instr = bnorm2instr_bp(instr, dir_name, filename, tensor_ids, O, O, OC, B, acc_obj)

		if isinstance(op, Convolution):
			data = op.input_tensors[0]
			weight = op.input_tensors[1]
			bias = op.input_tensors[2]
			out = op.output_tensors
			out_error = op.output_errors
			w_grad = op.weight_grads

			tensor_ids = (tensor_id[data.name], tensor_id[weight.name], tensor_id[bias.name], tensor_id[out.name])

			logger.debug('Input tensor: {}'.format(data.shape))
			logger.debug('Weight tensor: {}'.format(weight.shape))
			logger.debug('Bias tensor: {}'.format(bias.shape))
			logger.debug('Output tensor: {}'.format(out.shape))			
			logger.debug('out_error tensor: {}'.format(out_error.shape))
			logger.debug('w_grad tensor: {}'.format(w_grad.shape))	

			K = weight.fpga_shape[-2]
			O = out.fpga_shape[-2]
			S = op.stride[-1]
			IC = weight.fpga_shape[-1]
			OC = weight.fpga_shape[-4]
			B = data.fpga_shape[-4]
			logger.debug('K:{} O:{} S:{} IC:{} OC:{} B:{}'.format(K,O,S,IC,OC,B))

			#energy_cost and im2col are hardcoded
			energy_cost = (0,0,0,0,0,0,0,0,0,0)
			im2col = False
			conv_params = (acc_obj, K, O, S, IC, OC, B, prec, im2col, energy_cost)

			#error -> grads
			_, _, stats = optimize_for_order_bp(conv_params, sequential=False, pool_kernel=None, pool_stride=None)
			instr = conv2instr_bp(instr, stats, dir_name, filename+'_grad', tensor_ids, O, O, OC, IC, B, K, S, acc_obj, activation=None)

			#error -> prev. error
			_, _, stats = optimize_for_order(conv_params, sequential=False, pool_kernel=None, pool_stride=None)
			instr = conv2instr(instr, stats, dir_name, filename+'_bp', tensor_ids, O, O, OC, IC, B, K, S, acc_obj, None)

		if isinstance(op, Fork):
			O = op.output_tensors.fpga_shape[-2]
			OC = op.output_tensors.fpga_shape[-1]
			B = op.output_tensors.fpga_shape[-4]

			tensor_ids = (tensor_id[op.input_tensors[0].name], tensor_id[op.input_tensors[1].name], tensor_id[op.output_tensors.name])
			print('Writing to file: {}'.format(filename))
			#Add errors due to residual branch
			instr = eltwise_instr(instr, dir_name, filename, tensor_ids, O, O, OC, B, acc_obj, vector_op='Add', activation=None)


