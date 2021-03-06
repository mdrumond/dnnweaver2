class OPCodes:
    DRAM_RD     = 0
    DRAM_WR     = 1
    DRAM_RD_TP  = 2
    MAC       	= 3
    MULT       	= 4
    MULT_DER  	= 13
    ADD   	= 5
    ACT   	= 6
    POOL        = 7
    POOL_BP   	= 8
    BNORM   	= 9
    BNORM_BP    = 10
    NOP   	= 11
    EOL   	= 12

class ACTType:
    RELU     = 0
    SOFTMAX  = 1
    SIGMOID  = 2
    TANH     = 3
    RELU_DER = 4
    SOFTMAX_DER  = 5
    SIGMOID_DER  = 6
    TANH_DER     = 7    
    SOFTMAX_INV  = 8
    SIGMOID_INV  = 9
    TANH_INV     = 10    

class MACType:
    CONV     = 0
    FC	     = 1
