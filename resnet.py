from dnnweaver2.simulator.accelerator import Accelerator
from compiler_helpers import conv, fc, conv_instr_writer, fc_instr_writer, eltwise_instr_writer, batchnorm_instr_writer

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

B = 4

######### conv_1 ############
print('############### LAYER: conv1')

# "data" -> "pool1"
IH, IW, IC, K, OC, S = (224,224,3,7,64,2) #input_height, input_width, input_depth, batch_size, filter_size, output_depth, stride
stats = conv(IH, IW, IC, B, K, OC, S, acc_obj)
conv_instr_writer(stats, 'resnet50/', 'conv1', IH, IW, IC, B, K, OC, S, num_rows, num_cols, DRAM_BW, prec, None, False, None)
batchnorm_instr_writer('resnet50/', 'conv1_bn', 112, 112, OC, B, oprec, num_rows, num_cols, 'RELU', True, (2,2))

######### conv_2 ############

OC_prev = 64
IH_next = 56
IW_next = 56


# "pool1" -> "res2a_branch1"
IH, IW, IC, K, OC, S = (IH_next,IW_next,OC_prev,1,256,1) #input_height, input_width, input_depth, batch_size, filter_size, output_depth, stride
stats = conv(IH, IW, IC, B, K, OC, S, acc_obj)
conv_instr_writer(stats, 'resnet50/', 'conv2a1', IH, IW, IC, B, K, OC, S, num_rows, num_cols, DRAM_BW, prec, None, False, None)
batchnorm_instr_writer('resnet50/', 'conv2a1_bn', IH, IW, OC, B, oprec, num_rows, num_cols, None, False, None)


for i in ['a','b','c']:
	print('############### LAYER: conv_2'+i)
	# "pool1" -> "res2a_branch2a"
	IH, IW, IC, K, OC, S = (IH_next,IW_next,OC_prev,1,64,1) #input_height, input_width, input_depth, batch_size, filter_size, output_depth, stride
	stats = conv(IH, IW, IC, B, K, OC, S, acc_obj)
	conv_instr_writer(stats, 'resnet50/', 'conv2'+i+'2a', IH, IW, IC, B, K, OC, S, num_rows, num_cols, DRAM_BW, prec, None, False, None)
	batchnorm_instr_writer('resnet50/', 'conv2'+i+'2a_bn', IH, IW, OC, B, oprec, num_rows, num_cols, 'RELU', False, None)

	# "res2a_branch2a" -> "res2a_branch2b"
	IH, IW, IC, K, OC, S = (IH_next,IW_next,64,3,64,1) #input_height, input_width, input_depth, batch_size, filter_size, output_depth, stride
	stats = conv(IH, IW, IC, B, K, OC, S, acc_obj)
	conv_instr_writer(stats, 'resnet50/', 'conv2'+i+'2b', IH, IW, IC, B, K, OC, S, num_rows, num_cols, DRAM_BW, prec, None, False, None)
	batchnorm_instr_writer('resnet50/', 'conv2'+i+'2b_bn', IH, IW, OC, B, oprec, num_rows, num_cols, 'RELU', False, None)

	# "res2a_branch2b" -> "res2a_branch2c"
	IH, IW, IC, K, OC, S = (IH_next,IW_next,64,1,256,1) #input_height, input_width, input_depth, batch_size, filter_size, output_depth, stride
	stats = conv(IH, IW, IC, B, K, OC, S, acc_obj)
	conv_instr_writer(stats, 'resnet50/', 'conv2'+i+'2c', IH, IW, IC, B, K, OC, S, num_rows, num_cols, DRAM_BW, prec, None, False, None)
	batchnorm_instr_writer('resnet50/', 'conv2'+i+'2c_bn', IH, IW, OC, B, oprec, num_rows, num_cols, None, False, None)

	# "res2a_branch1" + "res2a_branch2c" -> "res2a"
	eltwise_instr_writer('resnet50/', 'res2'+i,IH_next,IW_next,256,B,oprec,num_rows,num_cols,'RELU',False,None,'Add')
	
	OC_prev = 256


######### conv_3 ############

OC_prev = 256
IH_next = 28
IW_next = 28

# "pool1" -> "res2a_branch1"
IH, IW, IC, K, OC, S = (IH_next,IW_next,OC_prev,1,512,1) #input_height, input_width, input_depth, batch_size, filter_size, output_depth, stride
stats = conv(IH, IW, IC, B, K, OC, S, acc_obj)
conv_instr_writer(stats, 'resnet50/', 'conv3a1', IH, IW, IC, B, K, OC, S, num_rows, num_cols, DRAM_BW, prec, None, False, None)
batchnorm_instr_writer('resnet50/', 'conv3a1_bn', IH, IW, OC, B, oprec, num_rows, num_cols, None, False, None)

for i in ['a','b','c','d']:
	print('############### LAYER: conv_3'+i)
	# "pool1" -> "res2a_branch2a"
	IH, IW, IC, K, OC, S = (IH_next,IW_next,OC_prev,1,128,1) #input_height, input_width, input_depth, batch_size, filter_size, output_depth, stride
	stats = conv(IH, IW, IC, B, K, OC, S, acc_obj)
	conv_instr_writer(stats, 'resnet50/', 'conv3'+i+'2a', IH, IW, IC, B, K, OC, S, num_rows, num_cols, DRAM_BW, prec, None, False, None)
	batchnorm_instr_writer('resnet50/', 'conv3'+i+'2a_bn', IH, IW, OC, B, oprec, num_rows, num_cols, 'RELU', False, None)

	# "res2a_branch2a" -> "res2a_branch2b"
	IH, IW, IC, K, OC, S = (IH_next,IW_next,OC,3,128,1) #input_height, input_width, input_depth, batch_size, filter_size, output_depth, stride
	stats = conv(IH, IW, IC, B, K, OC, S, acc_obj)
	conv_instr_writer(stats, 'resnet50/', 'conv3'+i+'2b', IH, IW, IC, B, K, OC, S, num_rows, num_cols, DRAM_BW, prec, None, False, None)
	batchnorm_instr_writer('resnet50/', 'conv3'+i+'2b_bn', IH, IW, OC, B, oprec, num_rows, num_cols, 'RELU', False, None)

	# "res2a_branch2b" -> "res2a_branch2c"
	IH, IW, IC, K, OC, S = (IH_next,IW_next,OC,1,512,1) #input_height, input_width, input_depth, batch_size, filter_size, output_depth, stride
	stats = conv(IH, IW, IC, B, K, OC, S, acc_obj)
	conv_instr_writer(stats, 'resnet50/', 'conv3'+i+'2c', IH, IW, IC, B, K, OC, S, num_rows, num_cols, DRAM_BW, prec, None, False, None)
	batchnorm_instr_writer('resnet50/', 'conv3'+i+'2c_bn', IH, IW, OC, B, oprec, num_rows, num_cols, None, False, None)

	# "res2a_branch1" + "res2a_branch2c" -> "res2a"
	eltwise_instr_writer('resnet50/', 'res3'+i,IH_next,IW_next,512,B,oprec,num_rows,num_cols,'RELU',False,None,'Add')
	
	OC_prev = 512

######### conv_4 ############

OC_prev = 512
IH_next = 14
IW_next = 14

# "pool1" -> "res2a_branch1"
IH, IW, IC, K, OC, S = (IH_next,IW_next,OC_prev,1,1024,1) #input_height, input_width, input_depth, batch_size, filter_size, output_depth, stride
stats = conv(IH, IW, IC, B, K, OC, S, acc_obj)
conv_instr_writer(stats, 'resnet50/', 'conv4a1', IH, IW, IC, B, K, OC, S, num_rows, num_cols, DRAM_BW, prec, None, False, None)
batchnorm_instr_writer('resnet50/', 'conv4a1_bn', IH, IW, OC, B, oprec, num_rows, num_cols, None, False, None)

for i in ['a','b','c','d','e','f']:
	print('############### LAYER: conv_4'+i)
	# "pool1" -> "res2a_branch2a"
	IH, IW, IC, K, OC, S = (IH_next,IW_next,OC_prev,1,256,1) #input_height, input_width, input_depth, batch_size, filter_size, output_depth, stride
	stats = conv(IH, IW, IC, B, K, OC, S, acc_obj)
	conv_instr_writer(stats, 'resnet50/', 'conv4'+i+'2a', IH, IW, IC, B, K, OC, S, num_rows, num_cols, DRAM_BW, prec, None, False, None)
	batchnorm_instr_writer('resnet50/', 'conv4'+i+'2a_bn', IH, IW, OC, B, oprec, num_rows, num_cols, 'RELU', False, None)

	# "res2a_branch2a" -> "res2a_branch2b"
	IH, IW, IC, K, OC, S = (IH_next,IW_next,OC,3,256,1) #input_height, input_width, input_depth, batch_size, filter_size, output_depth, stride
	stats = conv(IH, IW, IC, B, K, OC, S, acc_obj)
	conv_instr_writer(stats, 'resnet50/', 'conv4'+i+'2b', IH, IW, IC, B, K, OC, S, num_rows, num_cols, DRAM_BW, prec, None, False, None)
	batchnorm_instr_writer('resnet50/', 'conv4'+i+'2b_bn', IH, IW, OC, B, oprec, num_rows, num_cols, 'RELU', False, None)

	# "res2a_branch2b" -> "res2a_branch2c"
	IH, IW, IC, K, OC, S = (IH_next,IW_next,OC,1,1024,1) #input_height, input_width, input_depth, batch_size, filter_size, output_depth, stride
	stats = conv(IH, IW, IC, B, K, OC, S, acc_obj)
	conv_instr_writer(stats, 'resnet50/', 'conv4'+i+'2c', IH, IW, IC, B, K, OC, S, num_rows, num_cols, DRAM_BW, prec, None, False, None)
	batchnorm_instr_writer('resnet50/', 'conv4'+i+'2c_bn', IH, IW, OC, B, oprec, num_rows, num_cols, None, False, None)

	# "res2a_branch1" + "res2a_branch2c" -> "res2a"
	eltwise_instr_writer('resnet50/', 'res4'+i,IH_next,IW_next,1024,B,oprec,num_rows,num_cols,'RELU',False,None,'Add')
	
	OC_prev = 1024

######### conv_5 ############

OC_prev = 1024
IH_next = 7
IW_next = 7


# "pool1" -> "res2a_branch1"
IH, IW, IC, K, OC, S = (IH_next,IW_next,OC_prev,1,2048,1) #input_height, input_width, input_depth, batch_size, filter_size, output_depth, stride
stats = conv(IH, IW, IC, B, K, OC, S, acc_obj)
conv_instr_writer(stats, 'resnet50/', 'conv5a1', IH, IW, IC, B, K, OC, S, num_rows, num_cols, DRAM_BW, prec, None, False, None)
batchnorm_instr_writer('resnet50/', 'conv5a1_bn', IH, IW, OC, B, oprec, num_rows, num_cols, None, False, None)

for i in ['a','b','c']:
	print('############### LAYER: conv_5'+i)
	# "pool1" -> "res2a_branch2a"
	IH, IW, IC, K, OC, S = (IH_next,IW_next,OC_prev,1,512,1) #input_height, input_width, input_depth, batch_size, filter_size, output_depth, stride
	stats = conv(IH, IW, IC, B, K, OC, S, acc_obj)
	conv_instr_writer(stats, 'resnet50/', 'conv5'+i+'2a', IH, IW, IC, B, K, OC, S, num_rows, num_cols, DRAM_BW, prec, None, False, None)
	batchnorm_instr_writer('resnet50/', 'conv5'+i+'2a_bn', IH, IW, OC, B, oprec, num_rows, num_cols, 'RELU', False, None)

	# "res2a_branch2a" -> "res2a_branch2b"
	IH, IW, IC, K, OC, S = (IH_next,IW_next,OC,3,512,1) #input_height, input_width, input_depth, batch_size, filter_size, output_depth, stride
	stats = conv(IH, IW, IC, B, K, OC, S, acc_obj)
	conv_instr_writer(stats, 'resnet50/', 'conv5'+i+'2b', IH, IW, IC, B, K, OC, S, num_rows, num_cols, DRAM_BW, prec, None, False, None)
	batchnorm_instr_writer('resnet50/', 'conv5'+i+'2b_bn', IH, IW, OC, B, oprec, num_rows, num_cols, 'RELU', False, None)

	# "res2a_branch2b" -> "res2a_branch2c"
	IH, IW, IC, K, OC, S = (IH_next,IW_next,OC,1,512,1) #input_height, input_width, input_depth, batch_size, filter_size, output_depth, stride
	stats = conv(IH, IW, IC, B, K, OC, S, acc_obj)
	conv_instr_writer(stats, 'resnet50/', 'conv5'+i+'2c', IH, IW, IC, B, K, OC, S, num_rows, num_cols, DRAM_BW, prec, None, False, None)
	batchnorm_instr_writer('resnet50/', 'conv5'+i+'2c_bn', IH, IW, OC, B, oprec, num_rows, num_cols, None, False, None)

	# "res2a_branch1" + "res2a_branch2c" -> "res2a"	
	if i is 'c':
		eltwise_instr_writer('resnet50/', 'res5'+i,IH_next,IW_next,2048,B,oprec,num_rows,num_cols,'RELU',True,(7,7),'Add')
	else:
		eltwise_instr_writer('resnet50/', 'res5'+i,IH_next,IW_next,2048,B,oprec,num_rows,num_cols,'RELU',False,None,'Add')

	OC_prev = 2048

######### fc ############
print('############### LAYER: fc')

stats = fc(B, 2048, 1000, acc_obj)
fc_instr_writer(stats, 'resnet50/', 'fc', 2048, B, 1000, num_rows, num_cols, DRAM_BW, prec, 'SOFT')










