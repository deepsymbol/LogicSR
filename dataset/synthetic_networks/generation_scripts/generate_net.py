from methods.methods_plain import *
from tools.vis_net import vis_aig
from multiprocessing import Pool
import time
from tqdm import tqdm
from math import ceil
import os
import os.path as osp
import argparse


def single_aig_worker(args):
    star_time = time.time()
    k, l, Mstep, use_aig = args

    random.seed(time.time() + os.getpid())
    aig = random_logic_net(k, l, Mstep, use_aig, p0=3)
    aig.compute_truth_tables()
    key = aig.calculate_key()

    return key,aig

def single_write(args):
    '''
    write content to file
    '''
    use_aig, output_dir,tt_dir,i,aon,write_bnet,write_aag,write_truth, add_noise = args
    bnet_path = osp.join(output_dir, 'bnet', f"{i}.bnet")
    aag_path = osp.join(output_dir, 'aag', f"{i}.aag")
    filename = osp.join(output_dir, 'pic', f"{i}")
    vis_aig(aon, filename=filename,view=False, fmt="html")

    if write_bnet:
        #------------------------------------------#
        with open(bnet_path, 'w') as f:
            f.write("targets,factors\n")
            f.write(aon.to_bnet())
        #------------------------------------------#
    if write_aag:
        with open(aag_path, 'w') as f:
            f.write(aon.to_mid(use_aig)) # or flag: 1 -> OR; 0-> AND
    if write_truth:
        # add_noise = False
        noise_rates = [0,0.01,0.05] if add_noise else [0]
        for noise_rate in noise_rates:
            noise_tt_dir = osp.join(tt_dir,f'noise_{noise_rate}')
            os.makedirs(noise_tt_dir,exist_ok=True)
            truth_path = osp.join(noise_tt_dir,f"{i}.truth")
            aon.save_tt(truth_path,noise_rate=noise_rate)


def generate_unique_aigs(use_aig, k, l, Mstep, N, output_dir, write_aag=True,
                         write_bnet=False, write_truth=False,add_noise=False):
    # random.seed(1)
    aig_db = dict()
    while len(aig_db) < N:
        if len(aig_db) % 5 == 0:
            print(f"Generating {len(aig_db)}...")
        aig = random_logic_net(k, l, Mstep, use_aig)
        aig.compute_truth_tables()

        # calculate key
        key = aig.calculate_key()
        if key == 0:
            continue
        if key in aig_db:
            print("Key Exists")
            if len(aig_db[key].nodes) > len(aig.nodes):
                aig_db[key] = aig
        else:
            aig_db[key] = aig

    tt_dir = osp.join(output_dir, "tt")
    if write_bnet or write_truth:
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(tt_dir, exist_ok=True)
        os.makedirs(osp.join(output_dir,'bnet'),exist_ok=True)
        os.makedirs(osp.join(output_dir,'aag'),exist_ok=True)
    else:
        return list(aig_db.values())

    write_args = [
        (use_aig, output_dir, tt_dir, i, aon, write_bnet, write_aag, write_truth,add_noise)
        for i, (_, aon) in enumerate(aig_db.items(), 1)
    ]
    for arg in write_args:
        single_write(arg)


def generate_unique_aigs_parallel(use_aig, k, l, Mstep, N, output_dir = None,
                                  write_bnet=False, write_aag=True,
                                  write_truth=True, add_noise = True, multiple = 1.3, use_tqdm = False):
    aig_db = dict()
    max_layers = math.log2(Mstep / k) * 2

    with Pool() as pool:
        oversample = ceil(N * multiple)
        args = [(k, l, Mstep, use_aig) for _ in range(oversample)]
        results = list(tqdm(pool.imap(single_aig_worker, args), total=len(args),disable=True))
        for key, aig in results:
            if len(aig.outs) != l:
                print('not enough outs')
            if key==0:
                continue
            if key in aig_db:
                if len(aig_db[key].nodes) > len(aig.nodes):
                    aig_db[key] = aig
            else:
                aig_db[key] = aig
            if len(aig_db) >= N:
                break

    if write_bnet or write_truth:
        os.makedirs(output_dir, exist_ok=True)
        tt_dir = osp.join(output_dir, "tt")
        pic_dir = osp.join(output_dir, 'pic')
        aag_dir = osp.join(output_dir, 'aag')
        os.makedirs(pic_dir, exist_ok=True)
        os.makedirs(tt_dir, exist_ok=True)
        os.makedirs(aag_dir,exist_ok=True)
    else:
        return list(aig_db.values())

    write_args = [
        (use_aig, output_dir, tt_dir, i, aon,write_bnet,write_aag,write_truth, add_noise)
        for i, (_, aon) in enumerate(aig_db.items(), 1)
    ]

    with Pool() as pool:
        list(tqdm(pool.imap_unordered(single_write, write_args), total=len(write_args)))


if __name__ == "__main__":
    N = 5
    combins = [(6, 6, 20)] #,(5, 5, 20),(5, 5, 40),(10, 10, 40),(10, 10, 80),(10, 10, 160)]

    parser = argparse.ArgumentParser(description="Process some integers.")
    parser.add_argument('--use-aig', action='store_true', default=False,
                        help='A boolean flag to indicate if AN/ANO is used')
    parser.add_argument('--fmt', action='store_true', default='html',
                        help='A str to indicate the fmt of saving visualizations')
    args = parser.parse_args()
    use_aig = args.use_aig
    folder = f'../AN' if use_aig else f'../ANO'
    os.makedirs(folder,exist_ok=True)
    print(f"[WARN] use_aig = {use_aig}")

    for k, l, Mstep in combins:
        out_dir = f"{folder}/in{k}_out{l}/and{Mstep}"
        os.makedirs(out_dir,exist_ok=True)
        if Mstep * 2 < k / 2 + 1 or l < k / 2:
            print(f"ERR: bad AN/ANO structure for in {k} out {l} and {Mstep}")
            continue

        generate_unique_aigs_parallel(use_aig, k, l, Mstep, N, out_dir,
                                      add_noise=False, multiple=3)

