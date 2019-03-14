# A clean start is always good. 
# do not use -all, that one will delete all std cell libraries
remove_design -designs 

# read library
set target_library \ {/dkits/tsmc/28nm/cln28hpm/stclib/9-track/Front_End/timing_power_noise/NLDM/tcbn28hpmbwp35_120a/tcbn28hpmbwp35ss0p81v125c.db}
set link_library \ {/dkits/tsmc/28nm/cln28hpm/stclib/9-track/Front_End/timing_power_noise/NLDM/tcbn28hpmbwp35_120a/tcbn28hpmbwp35ss0p81v125c.db}

# analyze the design

#*******************************************
#TODO: add extra source files here

analyze -f vhdl ./src/constants.vhd
analyze -f vhdl ./src/mac.vhd
analyze -f vhdl ./src/buffer.vhd
analyze -f vhdl ./src/pe.vhd
analyze -f vhdl ./src/pe_array2D.vhd

#*******************************************
# elaborate the design 
elaborate pe_array2D

# set delay constraints
create_clock "clk" -name "global_clk" -period 0.75

# now let's just compile and see what comes up 
compile

# get some useful report about area and timing   
report_qor
report_area
report_timing
#quit design compiler
#exit
