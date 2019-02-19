from dnnweaver2.simulator.accelerator import Accelerator
from compiler_helpers import conv, conv_gd, fc, fc_gd, conv_instr_writer, fc_instr_writer, fc_gd_instr_writer, eltwise_instr_writer, conv_gd_instr_writer, batchnorm_gd_instr_writer

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

######### fc ############
print('############### LAYER: fc')

#error -> grads
stats = fc_gd(B, 1000, 2048, acc_obj)
fc_gd_instr_writer(stats, 'resnet50_gd/', 'fc_gd', 1000, B, 2048, num_rows, num_cols, DRAM_BW, prec, None)

#error -> prev. error
stats = fc(B, 1000, 2048, acc_obj)
fc_instr_writer(stats, 'resnet50_gd/', 'fc_bp', 1000, B, 2048, num_rows, num_cols, DRAM_BW, prec, None)

#multiply error with act. derivation
eltwise_instr_writer('resnet50_gd/', 'fc_bp_mult', 1, 1, 2048, B,oprec,num_rows,num_cols,None,False,None,'Mult')


######### conv_5 ############
print('############### LAYER: conv5_gd')

#batchnorm conv5c3
batchnorm_gd_instr_writer('resnet50_gd/', 'res5c_bp_bnorm', 1, 1, 2048, B, oprec, num_rows, num_cols, pooling=True, pool_window=(7,7))

#Calculate grads
IH, IW, IC, K, OC, S = (7,7,2048,3,512,1) #input_height, input_width, input_depth, batch_size, filter_size, output_depth, stride

stats = conv_gd(IH, IW, IC, B, K, OC, S, acc_obj, pooling=False, pool_window=None)
conv_gd_instr_writer(stats, 'resnet50_gd/', 'conv5c2c_gd', IH, IW, IC, B, K, OC, S, num_rows, num_cols, DRAM_BW, prec, None, False, None)

#Propagate error
stats = conv(IH, IW, IC, B, K, OC, S, acc_obj)
conv_instr_writer(stats, 'resnet50_gd/', 'conv5c2c_bp', IH, IW, IC, B, K, OC, S, num_rows, num_cols, DRAM_BW, prec, None, False, None)

#multiply error with act. derivation
eltwise_instr_writer('resnet50_gd/', 'conv5c2c_bp_mult', IH, IW, OC, B,oprec,num_rows,num_cols,None,False,None,'Mult')


















