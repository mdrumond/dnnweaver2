import math
import functools
import time
import logging

from itertools import permutations
from multiprocessing import Pool, cpu_count

from dnnweaver2.utils.utils import ceil_a_by_b, log2
from dnnweaver2.simulator.loop_stack import LoopStack
from dnnweaver2.simulator.stats import Stats

import numpy as np

logger = logging.getLogger('{}.{}'.format(__name__, 'Optimizer'))
logger.setLevel(logging.ERROR)

class OP_TYPE:
    FW     = 0
    GD     = 1

tile_deps = {}
tile_deps['B/b']   = {'ibuf': True,  'wbuf': False, 'obuf': True,  'bbuf': False}
tile_deps['OW/ow'] = {'ibuf': True,  'wbuf': False, 'obuf': True,  'bbuf': False}
tile_deps['OH/oh'] = {'ibuf': True,  'wbuf': False, 'obuf': True,  'bbuf': False}
tile_deps['IC/ic'] = {'ibuf': True,  'wbuf': True,  'obuf': False, 'bbuf': False}
tile_deps['OC/oc'] = {'ibuf': False, 'wbuf': True,  'obuf': True,  'bbuf': True}
tile_deps['G/g'] = {'ibuf': False, 'wbuf': True,  'obuf': True,  'bbuf': True}


def optimize_for_order(conv_params, sequential=True, op_type=None):
    # Generate permutations for the order
    loops = ['B/b', 'OW/ow', 'OH/oh', 'IC/ic', 'OC/oc', 'G/g']
    order = set(permutations(loops))

    return_dict = {}
    acc_obj, K, O, S, IC, OC, B, G, energy_cost = conv_params

    if not sequential:
        _bound_optimizer_method = functools.partial(_optimize_for_order, conv_params, op_type)

        try:
            pool = Pool(cpu_count())
            results = pool.map_async(_bound_optimizer_method, order).get(10000)

            pool.close()
            pool.join()

            best_cycles = None
	    best_energy = None
            for r in results:
		tiling, order_type, cycles, energy = r
		
                if best_cycles is None or best_cycles > cycles or (best_cycles == cycles and best_energy >= energy):
                    best_cycles = cycles
		    best_energy = energy
                    best_tiling = tiling
                    best_order = order_type
	    	    #print('Order:{} Tiling:{} Cycles:{} Energy:{}'.format(order_type, tiling, cycles, energy))

	    stats = get_stats_fast(conv_params, op_type, best_tiling, best_order, verbose=True)
            return best_tiling, best_order, stats

        except KeyboardInterrupt:
            pool.terminate()
            pool.join()
            return

    else:
        best_cycles = None
        best_tiling = None
        best_order  = None
        for o in order:
            tiling, order_type, cycles,  energy = _optimize_for_order(conv_params, backprop, o)
            if best_cycles is None or best_cycles > cycles:
                best_cycles = cycles
                best_tiling = tiling
                best_order  = order_type

	stats = get_stats_fast(conv_params, best_tiling, best_order, verbose=True)

        return best_tiling, best_order, stats

def _optimize_for_order(conv_params, op_type, order_type, verbose=False, ):

    acc_obj, K, O, S, IC, OC, B, G, energy_cost = conv_params
    I = (O - 1) * S + K

    # We do not tile the "K" dimension and compute an entire 2-D conv at a time
    num_G_tiles = int(math.ceil(log2(G))) + 1
    num_O_tiles = int(math.ceil(log2(O))) + 1
    num_IC_tiles = int(math.ceil(log2(IC))) + 1
    num_OC_tiles = int(math.ceil(log2(OC))) + 1
    num_B_tiles = int(math.ceil(log2(B))) + 1

    best_cycles = None
    best_energy = None
    best_tiling = None

    cycle_array = np.zeros((num_B_tiles, num_O_tiles, num_IC_tiles, num_OC_tiles), dtype=np.float)
    energy_array = np.zeros((num_B_tiles, num_O_tiles, num_IC_tiles, num_OC_tiles), dtype=np.float)

    for _g in range(num_G_tiles):
	g = min(1 << _g, G)
	num_g = ceil_a_by_b(G, g)

        for _b in range(num_B_tiles):
		b = min(1 << _b, B)
		num_b = ceil_a_by_b(B, b)

		for _o in range(num_O_tiles):
		    ow = min(1 << _o, O)
		    oh = ow
		    num_ow = ceil_a_by_b(O, ow)
		    num_oh = ceil_a_by_b(O, oh)

		    if num_ow * ow != O:
			continue

		    for _ic in range(num_IC_tiles):
			ic = min(1 << _ic, IC)
			num_ic = ceil_a_by_b(IC, ic)

			for _oc in range(num_OC_tiles):
			    oc = min(1 << _oc, OC)
			    num_oc = ceil_a_by_b(OC, oc)

			    tiling = {}
			    tiling['B/b'] = (num_b, b)
			    tiling['OW/ow'] = (num_ow, ow)
			    tiling['OH/oh'] = (num_oh, oh)
			    tiling['IC/ic'] = (num_ic, ic)
			    tiling['OC/oc'] = (num_oc, oc)
			    tiling['G/g'] = (num_g, g)

			    stats = get_stats_fast(conv_params, op_type, tiling, order_type, verbose=verbose)

			    if stats is None:
				continue

			    cycles = stats.total_cycles
			    cycle_array[_b, _o, _ic, _oc] = cycles
			    mem_cycles = stats.mem_stall_cycles
			    energy = stats.get_energy(energy_cost)

			    if best_cycles is None or best_cycles > cycles or (best_cycles == cycles and best_energy > energy):
				best_energy = energy
				best_cycles = cycles
				best_mem_cycles = mem_cycles
				best_order = order_type
				best_tiling = tiling

			    	#print('Order:{} Tiling:{} Cycles:{:,} Energy:{:,}'.format(order_type,tiling,cycles,energy))

    return (best_tiling, order_type, best_cycles, best_energy)

def get_stats_fast(conv_params, op_type, tiling, order_type, verbose=False):
    acc_obj, K, O, S, IC, OC, B, G, energy_cost = conv_params

    num_b, b = tiling['B/b']
    num_ow, ow = tiling['OW/ow']
    num_oh, oh = tiling['OH/oh']
    num_ic, ic = tiling['IC/ic']
    num_oc, oc = tiling['OC/oc']
    num_g, g = tiling['G/g']

    kw = kh = K

    ih = (oh - 1) * S + kh
    iw = (ow - 1) * S + kw

    iprec, wprec, bprec, oprec = acc_obj.prec

    stats = Stats()
    stats.tiling = tiling
    stats.order = order_type

    if op_type == OP_TYPE.FW:
    	stats, writes, reads = memory_access_fw(stats, acc_obj, ih, iw, ic, kh, kw, oh, ow, oc, b, g, num_ic, num_oh, num_ow, num_oc, num_b, num_g)
    elif op_type == OP_TYPE.GD:
	stats, writes, reads = memory_access_gd(stats, acc_obj, ih, iw, ic, kh, kw, oh, ow, oc, b, g, num_ic, num_oh, num_ow, num_oc, num_b, num_g)
    else:
	assert False

    # Skip if overutilizing resources
    overflow = False
    for namespace in writes:
        if writes[namespace] > acc_obj.sram[namespace]/2:
            overflow = True
    if overflow:
        return

    max_write_size = {}
    max_read_size = {}
    for namespace in writes:
        max_write_size[namespace] = writes[namespace]
    for namespace in reads:
        max_read_size[namespace] = reads[namespace]

    rd_cache_hit = {'wbuf': True, 'ibuf': True, 'obuf': True, 'bbuf': True}
    wr_cache_hit = {'wbuf': True, 'obuf': True}
    if verbose:
        logger.debug('Initialize reads/writes')
        logger.debug('\tTiling: {}'.format(tiling))
        logger.debug('\tReads : {}'.format(reads))
        logger.debug('\tWrites: {}'.format(writes))

    for loop in order_type:
        num_tiles, tile_size = tiling[loop]
        for namespace in writes:
            if rd_cache_hit[namespace]:
                if tile_deps[loop][namespace]:
                    writes[namespace] *= num_tiles
                    rd_cache_hit[namespace] = False
            else:
                writes[namespace] *= num_tiles

        for namespace in reads:
            if wr_cache_hit[namespace]:
                if tile_deps[loop][namespace]:
                    reads[namespace] *= num_tiles
                    wr_cache_hit[namespace] = False
            else:
                reads[namespace] *= num_tiles

        if verbose:
            logger.debug('Loop: {}'.format(loop))
            logger.debug('\tLoop range: {}'.format(tiling[loop]))
            logger.debug('\tMax write size: {}'.format(max_write_size))
            logger.debug('\tMax read size: {}'.format(max_read_size))
            logger.debug('\tLoop Dependencies: {}'.format(tile_deps[loop]))
            logger.debug('\tLoop Promote: {}'.format(rd_cache_hit))
            logger.debug('\tReads : {}'.format(reads))
            logger.debug('\tWrites: {}'.format(writes))

    for namespace in writes:
        stats.writes[namespace] = writes[namespace]
        stats.reads['dram'] += writes[namespace]
    for namespace in reads:
        stats.reads[namespace] = reads[namespace]
        stats.writes['dram'] += reads[namespace]

    # TODO: update
    stats.initial_dram_reads = 0
    stats.final_dram_writes = 0
    for namespace in max_write_size:
        stats.initial_dram_reads += max_write_size[namespace]
    for namespace in max_read_size:
        stats.final_dram_writes += max_read_size[namespace]
    latency = acc_obj.get_mem_read_cycles('dram', stats.initial_dram_reads) + \
            acc_obj.get_mem_write_cycles('dram', stats.final_dram_writes)

    stats.total_dram_accesses = stats.reads['dram'] + stats.writes['dram']
    stats.middle_dram_accesses = stats.total_dram_accesses - stats.initial_dram_reads - stats.final_dram_writes

    if verbose:
	    logger.debug('Compute cycle per tile : {:>20,}'.format(acc_obj.get_compute_cycles(ic, oc, ow, oh, b, kw, kh)))
	    logger.debug('# of tiles : {:>20,}'.format(num_tiles))
    stats.compute_cycles = num_tiles * g * acc_obj.get_compute_cycles(ic, oc, ow, oh, b, kw, kh)
    stats.memory_cycles_required = ceil_a_by_b(stats.middle_dram_accesses, acc_obj.mem_if_width)

    memory_stalls = max(0, stats.memory_cycles_required - stats.compute_cycles) + latency
    stats.total_cycles = stats.compute_cycles + memory_stalls
    stats.mem_stall_cycles = memory_stalls

    if verbose:
        logger.debug('Compute cycles : {:>20,}'.format(stats.compute_cycles))
        logger.debug('Memory cycles  : {:>20,}'.format(stats.memory_cycles_required + latency))
        logger.debug('Memory stalls  : {:>20,}'.format(memory_stalls))
	logger.debug('Total_cycles  : {:>20,}'.format(stats.total_cycles))
    return stats



def memory_access_fw(stats, acc_obj, ih, iw, ic, kh, kw, oh, ow, oc, b, g, num_ic, num_oh, num_ow, num_oc, num_b, num_g, verbose=False):
    iprec, wprec, bprec, oprec = acc_obj.prec

    writes = {}
    reads = {}
    writes['wbuf'] = ceil_a_by_b(ic, acc_obj.N) * acc_obj.N * kh * kw * ceil_a_by_b(oc, acc_obj.M) * acc_obj.M * g * wprec
    writes['ibuf'] = iw * ih * ceil_a_by_b(ic, acc_obj.N) * acc_obj.N * b * iprec
    writes['bbuf'] = ceil_a_by_b(oc, acc_obj.M) * acc_obj.M * g * bprec
    writes['obuf'] = ow * oh * ceil_a_by_b(oc, acc_obj.M) * acc_obj.M * b * g * oprec
    
    reads['obuf'] = ow * oh * ceil_a_by_b(oc, acc_obj.M) * acc_obj.M * b * g * oprec

    is_loop = ceil_a_by_b(oc, acc_obj.M) * acc_obj.M * g
    os_loop = ceil_a_by_b(ic, acc_obj.N) * acc_obj.N * kh * kw
    ws_loop = b * oh * ow
    # Input Stationary energy
    # kw * kh * ic * oh * ow * b -> oc
    is_energy = (os_loop * ws_loop) * (iprec    + is_loop * (wprec + oprec))
    # Output Stationary energy
    # oc * oh * ow * b -> kw * kh * ic
    os_energy = (is_loop * ws_loop) * (oprec    + os_loop * (iprec + wprec))
    # Weight Stationary energy
    # kw * kh * ic * oc -> b * ow * oh
    ws_energy = (os_loop * is_loop) * (wprec    + ws_loop * (iprec + oprec))

    min_energy = min(is_energy, ws_energy, os_energy)
    num_tiles = num_g * num_b * num_ow * num_oh * num_ic * num_oc

    if is_energy == min_energy:
        if verbose:
            logger.debug('SRAM access order: Input Stationary')
        stats.reads['ibuf'] += num_tiles * (kw * kh * ic * oh * ow * b) * iprec
        stats.reads['obuf'] += num_tiles * (kw * kh * ic * oh * ow * b) * oc * g *oprec
        stats.writes['obuf'] += num_tiles * (kw * kh * ic * oh * ow * b) * oc * g * oprec
        stats.reads['wbuf'] += num_tiles * (kw * kh * ic * oh * ow * b) * oc * g * wprec

    elif os_energy == min_energy:
        if verbose:
            logger.debug('SRAM access order: Output Stationary')
        stats.reads['ibuf'] += num_tiles * (oc * oh * ow * b * g) * (kw * kh * ic) * iprec
        stats.reads['obuf'] += num_tiles * (oc * oh * ow * b * g) * oprec
        stats.writes['obuf'] += num_tiles * (oc * oh * ow * b * g) * oprec
        stats.reads['wbuf'] += num_tiles * (oc * oh * ow * b * g) * (kw * kh * ic) * wprec

    else:
        if verbose:
            logger.debug('SRAM access order: Weight Stationary')
        stats.reads['ibuf'] += num_tiles * (kw * kh * ic * oc * g) * (b * ow * oh) * iprec
        stats.reads['obuf'] += num_tiles * (kw * kh * ic * oc * g) * (b * ow * oh) * oprec
        stats.writes['obuf'] += num_tiles * (kw * kh * ic * oc * g) * (b * ow * oh) * oprec
        stats.reads['wbuf'] += num_tiles * (kw * kh * ic * oc * g) * wprec

    return stats, writes, reads

def memory_access_bp(stats, acc_obj, ih, iw, ic, kh, kw, oh, ow, oc, b, g, num_ic, num_oh, num_ow, num_oc, num_b, num_g, verbose=False):
    iprec, wprec, bprec, oprec = acc_obj.prec

    writes = {}
    reads = {}
    writes['wbuf'] = ceil_a_by_b(ic, acc_obj.N) * acc_obj.N * kh * kw * ceil_a_by_b(oc, acc_obj.M) * acc_obj.M * g * wprec
    writes['ibuf'] = iw * ih * ceil_a_by_b(ic, acc_obj.N) * acc_obj.N * b * iprec
    writes['bbuf'] = ceil_a_by_b(oc, acc_obj.M) * acc_obj.M * g * bprec
    writes['obuf'] = ow * oh * ceil_a_by_b(oc, acc_obj.M) * acc_obj.M * b * g * oprec
    
    reads['ibuf'] = iw * ih * ceil_a_by_b(ic, acc_obj.N) * acc_obj.N * b * iprec

    is_loop = ceil_a_by_b(oc, acc_obj.M) * acc_obj.M * g
    os_loop = ceil_a_by_b(ic, acc_obj.N) * acc_obj.N * kh * kw
    ws_loop = b * oh * ow
    # Input Stationary energy
    # kw * kh * ic * oh * ow * b -> oc
    is_energy = (os_loop * ws_loop) * (iprec    + is_loop * (wprec + oprec))
    # Output Stationary energy
    # oc * oh * ow * b -> kw * kh * ic
    os_energy = (is_loop * ws_loop) * (oprec    + os_loop * (iprec + wprec))
    # Weight Stationary energy
    # kw * kh * ic * oc -> b * ow * oh
    ws_energy = (os_loop * is_loop) * (wprec    + ws_loop * (iprec + oprec))

    min_energy = min(is_energy, ws_energy, os_energy)
    num_tiles = num_g * num_b * num_ow * num_oh * num_ic * num_oc

    if is_energy == min_energy:
        if verbose:
            logger.debug('SRAM access order: Input Stationary')
        stats.reads['ibuf'] += num_tiles * (kw * kh * ic * oh * ow * b) * iprec
        stats.reads['obuf'] += num_tiles * (kw * kh * ic * oh * ow * b) * oc * g *oprec
        stats.writes['obuf'] += num_tiles * (kw * kh * ic * oh * ow * b) * oc * g * oprec
        stats.reads['wbuf'] += num_tiles * (kw * kh * ic * oh * ow * b) * oc * g * wprec

    elif os_energy == min_energy:
        if verbose:
            logger.debug('SRAM access order: Output Stationary')
        stats.reads['ibuf'] += num_tiles * (oc * oh * ow * b * g) * (kw * kh * ic) * iprec
        stats.reads['obuf'] += num_tiles * (oc * oh * ow * b * g) * oprec
        stats.writes['obuf'] += num_tiles * (oc * oh * ow * b * g) * oprec
        stats.reads['wbuf'] += num_tiles * (oc * oh * ow * b * g) * (kw * kh * ic) * wprec

    else:
        if verbose:
            logger.debug('SRAM access order: Weight Stationary')
        stats.reads['ibuf'] += num_tiles * (kw * kh * ic * oc * g) * (b * ow * oh) * iprec
        stats.reads['obuf'] += num_tiles * (kw * kh * ic * oc * g) * (b * ow * oh) * oprec
        stats.writes['obuf'] += num_tiles * (kw * kh * ic * oc * g) * (b * ow * oh) * oprec
        stats.reads['wbuf'] += num_tiles * (kw * kh * ic * oc * g) * wprec

    return stats, writes, reads


def memory_access_gd(stats, acc_obj, ih, iw, ic, kh, kw, oh, ow, oc, b, g, num_ic, num_oh, num_ow, num_oc, num_b, num_g, verbose=False):
    iprec, wprec, bprec, oprec = acc_obj.prec

    writes = {}
    reads = {}

    writes['wbuf'] =  ceil_a_by_b(ic, acc_obj.N) * acc_obj.N * kh * kw * ceil_a_by_b(oc, acc_obj.M) * acc_obj.M * g * wprec
    writes['ibuf'] = iw * ih * ceil_a_by_b(ic, acc_obj.N) * acc_obj.N * b * iprec
    writes['obuf'] = ow * oh * ceil_a_by_b(oc, acc_obj.M) * acc_obj.M * b * g * oprec

    reads['wbuf'] = ceil_a_by_b(ic, acc_obj.N) * acc_obj.N * kh * kw * ceil_a_by_b(oc, acc_obj.M) * acc_obj.M * g * wprec

    is_loop = ceil_a_by_b(oc, acc_obj.M) * acc_obj.M * g
    os_loop = ceil_a_by_b(ic, acc_obj.N) * acc_obj.N * kh * kw
    ws_loop = b * oh * ow
    # Input Stationary energy
    # kw * kh * ic * oh * ow * b -> oc
    is_energy = (os_loop * ws_loop) * (iprec    + is_loop * (wprec + oprec))
    # Output Stationary energy
    # oc * oh * ow * b -> kw * kh * ic
    os_energy = (is_loop * ws_loop) * (oprec    + os_loop * (iprec + wprec))
    # Weight Stationary energy
    # kw * kh * ic * oc -> b * ow * oh
    ws_energy = (os_loop * is_loop) * (wprec    + ws_loop * (iprec + oprec))

    min_energy = min(is_energy, ws_energy, os_energy)

    num_tiles = num_g * num_b * num_ow * num_oh * num_ic * num_oc

    if is_energy == min_energy:
        if verbose:
            logger.debug('SRAM access order: Input Stationary')
        stats.reads['ibuf'] += num_tiles * (kw * kh * ic * oh * ow * b) * iprec
        stats.reads['obuf'] += num_tiles * (kw * kh * ic * oh * ow * b) * oc * g * oprec
        stats.writes['wbuf'] += num_tiles * (kw * kh * ic * oh * ow * b) * oc * g * wprec
        stats.reads['wbuf'] += num_tiles * (kw * kh * ic * oh * ow * b) * oc * g * wprec

    elif os_energy == min_energy:
        if verbose:
            logger.debug('SRAM access order: Output Stationary')
        stats.reads['ibuf'] += num_tiles * (oc * oh * ow * b * g) * (kw * kh * ic) * iprec
        stats.reads['obuf'] += num_tiles * (oc * oh * ow * b * g) * oprec
        stats.writes['wbuf'] += num_tiles * (oc * oh * ow * b * g) * (kw * kh * ic) * wprec
        stats.reads['wbuf'] += num_tiles * (oc * oh * ow * b * g) * (kw * kh * ic) * wprec

    else:
        if verbose:
            logger.debug('SRAM access order: Weight Stationary')
        stats.reads['ibuf'] += num_tiles * (kw * kh * ic * oc * g) * (b * ow * oh) * iprec
        stats.reads['obuf'] += num_tiles * (kw * kh * ic * oc * g) * (b * ow * oh) * oprec
        stats.writes['wbuf'] += num_tiles * (kw * kh * ic * oc * g) * wprec
	stats.reads['wbuf'] += num_tiles * (kw * kh * ic * oc * g) * wprec

    return stats, writes, reads

