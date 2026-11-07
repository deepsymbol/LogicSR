from methods.methods_merge import *
from generate_net import generate_unique_aigs_parallel, allocate_output

import argparse
import gc
import os.path as osp
from tools.vis_net import vis_aig
from multiprocessing import Pool


def find_max_prefix_number_dir(directory, suffix):
    '''
    find file names in a dir and decide whether to preserve the previous results
    '''
    max_number = 0
    for filename in os.listdir(directory):
        if filename.endswith(suffix):
            number = int(filename.split('.')[0])
            if number > max_number:
                max_number = number
    return max_number

def write_to_file(max_num, out_dir, combined_aig, USE_AIG = False):
    '''
    write content to file
    '''
    os.makedirs(osp.join(out_dir,'aag'),exist_ok=True)
    aag_path = osp.join(out_dir,'aag', f"{max_num + 1}.aag")
    with open(aag_path, 'w') as f:
        aag_data = combined_aig.to_mid(use_aig=USE_AIG)
        f.write(aag_data)

def loop_aigs(aigs,max_num,num_new_in):
    num = num_new_in
    loop_each_combin = (2**num_new_in)
    final_len = 2**max_num
    for aig in reversed(aigs):
        num += aig.k
        aig.loop_truth(loop_each_combin,final_len)
        loop_each_combin *= (2**aig.k)

def save_files(arg):
    i, new_aig, out_dir = arg
    filename = osp.join(out_dir, 'pic', f'{i+1}')
    vis_aig(new_aig, filename, fmt='html')
    write_to_file(i, out_dir, new_aig, USE_AIG=False)
    tt_path = osp.join(out_dir, "tt", f"{i+1}.truth")
    new_aig.save_tt(tt_path)
    del new_aig

def merged_worker(args):
    i, aigs, num_in, num_out, in_left, out_left, Msteps, tt_size, out_dir, USE_AIG = args
    # aigs = [aig.copy() for aig in aags]
    new_aig = generate_merged_aig(
        USE_AIG,aigs, num_in, num_out,
        in_left, out_left, Msteps,
        tt_size=tt_size, use_tqdm=False
    )
    start_id = len(new_aig.nodes) - Msteps
    allocate_output(new_aig, out_left=num_out, start_id=num_in, clear_outs= True)
    if len(new_aig.outs)!=num_out:
        print(f'number of out exceeds planned: {len(new_aig.outs)}')

    return new_aig

def move_aigs(aigs,num_in,max_num,in_left):
    idx = num_in - aigs[0].k
    inidx = 0
    for aig in aigs:
        aig.compute_truth_tables()
        aig.increment_ids(incre=idx, in_incre=inidx)
        inidx += aig.k
        idx += len(aig.nodes) - aig.k

    loop_aigs(aigs, max_num, in_left)


def setup_output_dir(USE_AIG, num_in, num_out, tt_len_exp, num_ands, rewrite):
    """Create output directories and return output path + index."""
    folder = '../AN' if USE_AIG else '../ANO'
    out_dir = f"{folder}/in{num_in}_out{num_out}_tt{tt_len_exp}"
    os.makedirs(out_dir, exist_ok=True)
    out_dir = osp.join(out_dir, f'and{num_ands}')
    os.makedirs(out_dir, exist_ok=True)

    suf_dir = osp.join(out_dir, 'tt')
    os.makedirs(suf_dir, exist_ok=True)
    os.makedirs(osp.join(out_dir, 'pic'), exist_ok=True)
    index = 0 if rewrite else find_max_prefix_number_dir(suf_dir, suffix='truth')

    return out_dir, index

def main_func(args,combins,small_aigs):
    USE_AIG = args.use_aig
    parallel = args.parallel
    rewrite = args.rewrite
    num_new_aigs = args.num_aigs
    tt_len_exp = args.tt_len
    batch_size = args.batch_size
    multiple = args.multiple

    for combin,small_aig in zip(combins,small_aigs):
        num_in,num_out,num_ands = combin
        k, l ,N, and_gates = small_aig
        in_left = num_in - k * N
        out_left = num_out - l * N
        out_dir,index = setup_output_dir(USE_AIG, num_in, num_out, tt_len_exp, num_ands, rewrite)

        aigs = generate_unique_aigs_parallel(USE_AIG,k,l,and_gates,N,write_truth=False,multiple=multiple)
        Msteps = num_ands - sum([len(aig.nodes)-aig.k for aig in aigs])
        max_num = min(num_in, tt_len_exp)
        move_aigs(aigs,num_in,max_num,in_left)

        if parallel:
            batch = []
            for i in tqdm(range(num_new_aigs), total=num_new_aigs, desc="Generating + Saving"):
                aigs = generate_unique_aigs_parallel(USE_AIG, k, l, and_gates, N, write_truth=False, multiple=3)
                move_aigs(aigs, num_in, max_num, in_left)

                args = (index + i, aigs, num_in, num_out, in_left, out_left, Msteps, 2 ** max_num, out_dir, USE_AIG)
                result = merged_worker(args)
                batch.append((args[0], result, out_dir))
                del aigs

                if len(batch) == batch_size or i == num_new_aigs - 1:
                    with Pool() as pool:
                        pool.map(save_files, batch)

                    batch.clear()
                    gc.collect()
        else:
            args_list = [(index + i, aigs, num_in, num_out, in_left, out_left, Msteps,
                          2 ** max_num, out_dir, USE_AIG) for i in range(num_new_aigs)]
            for args in tqdm(args_list,total=len(args_list)):
                new_aig = merged_worker(args)
                arg = (args[0], new_aig, out_dir)
                save_files(arg)
                gc.collect()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Batch AIG Generator with small_aig combinations")
    parser.add_argument('--use-aig', action='store_true', default=False)
    parser.add_argument('--num_aigs', type=int, default=3)
    parser.add_argument('--tt_len', type=int, default=15)
    parser.add_argument('--rewrite', action='store_true', default=True)
    parser.add_argument('--parallel', action='store_true', default=True)
    parser.add_argument('--batch_size', type=int, default=6)
    parser.add_argument('--multiple', type=int, default=5)
    args = parser.parse_args()

    combins,small_aigs= ([(20,20,300)],[(7,7,2,120)])

    # small_aigs = generate_small_aig_combinations(num_in,num_out,num_ands)

    main_func(args,combins,small_aigs)

