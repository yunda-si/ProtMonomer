import os
from collections import deque
from Bio import SeqIO
import random
import argparse

def read_a2m(a2m_file, min_cov=0):
    msa_data = []
    seq = 'TEMP'
    gap_cov = 1-min_cov
    header = deque(['',''],maxlen=2)

    with open(a2m_file,'rb',buffering=8192) as f:
        for row in f:
            row = bytes.decode(row).strip()
            if row.startswith('>'):
                lenseq = len(seq)
                header.append(row)
                if seq.count('-')<lenseq*gap_cov:
                    msa_data.append((header.popleft(),seq))
                seq = ''
            else:
                seq += row
        lenseq = len(seq)
        if seq.count('-')<lenseq*gap_cov:
            msa_data.append((header.popleft(),seq))

    f.close()
    return msa_data[1:]

def save_a3m(a3m_file, msa_data):

    with open(a3m_file, 'w') as f:
        for header, seq in msa_data:
            f.write(f'{header}\n')
            f.write(f'{seq}\n')
    f.close()


bfd_path = '/share/home/yunda_si/database/BFD/bfd_metaclust_clu_complete_id30_c90_final_seq.sorted_opt'
uniref90_path = "/share/home/yunda_si/database/uniref90.fasta"
mgnify_path = "/share/home/yunda_si/database/mgy_clusters.fa"

def search_msa(final_msa_path, seq_file):

    if seq_file.count('.fasta') != 1:
        raise

    if not os.path.exists(final_msa_path):
        os.makedirs(final_msa_path)

    seq_id = os.path.basename(seq_file).split('.fasta')[0]

    key = 'bfd'
    msa_file = os.path.join(final_msa_path, f'{seq_id}_{key}.a3m')
    os.system(f'hhblits -i {seq_file} -n 3 -realign_max 100000 -maxfilt 100000 -min_prefilter_hits 1000 -cpu 2 -oa3m {msa_file} -d {bfd_path} >/dev/null 2>&1')

    key = 'uniref90'
    msa_file = os.path.join(final_msa_path, f'{seq_id}_{key}.a3m')
    sto_file = os.path.join(final_msa_path, f'{seq_id}_{key}.sto')
    masksto_file = os.path.join(final_msa_path, f'{seq_id}_{key}.masto')
    a2m_file = os.path.join(final_msa_path, f'{seq_id}_{key}.a2m')

    os.system(f'jackhmmer -N 1 -E 0.0001 --incE 0.0001 --F1 0.0005 --F2 0.00005 --F3 0.0000005 --cpu 1 --noali -A {sto_file} {seq_file} {uniref90_path} >/dev/null 2>&1')
    os.system(f'esl-alimask --rf-is-mask {sto_file}>{masksto_file}')
    os.system(f'esl-reformat a2m {masksto_file}>{a2m_file}')
    save_a3m(msa_file, read_a2m(a2m_file))
    os.remove(masksto_file)
    os.remove(a2m_file)
    os.remove(sto_file)

    key = 'mgnify'
    msa_file = os.path.join(final_msa_path, f'{seq_id}_{key}.a3m')
    sto_file = os.path.join(final_msa_path, f'{seq_id}_{key}.sto')
    masksto_file = os.path.join(final_msa_path, f'{seq_id}_{key}.masto')
    a2m_file = os.path.join(final_msa_path, f'{seq_id}_{key}.a2m')

    os.system(f'jackhmmer -N 1 -E 0.0001 --incE 0.0001 --F1 0.0005 --F2 0.00005 --F3 0.0000005 --cpu 1 --noali -A {sto_file} {seq_file} {mgnify_path}  >/dev/null 2>&1')
    os.system(f'esl-alimask --rf-is-mask {sto_file}>{masksto_file}')
    os.system(f'esl-reformat a2m {masksto_file}>{a2m_file}')
    save_a3m(msa_file, read_a2m(a2m_file))
    os.remove(masksto_file)
    os.remove(a2m_file)
    os.remove(sto_file)

    fasta_seq = open(seq_file).readlines()[-1].strip()
    len_seq = len(fasta_seq)
    all_seq = set()
    for key in ['bfd', 'uniref90', 'mgnify']:
        msa_file = os.path.join(final_msa_path, f'{seq_id}_{key}.a3m')
        if not os.path.exists(msa_file):
            continue

        parsed_msa = SeqIO.parse(msa_file, 'fasta')
        for record in parsed_msa:
            aligned_seq = ''.join([aa for aa in record if not aa.islower()])
            if len(aligned_seq) != len_seq:
                pass
            else:
                all_seq.add(aligned_seq)

    all_seq = list(all_seq)
    random.shuffle(all_seq)
    all_seq.insert(0, fasta_seq)

    msa_file = os.path.join(final_msa_path, f'{seq_id}.a3m')

    with open(os.path.join(msa_file), 'w') as f:
        for idx, seq in enumerate(all_seq[:50000]):
            f.write(f'>{idx}\n')
            f.write(f'{seq}\n')

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Search MSA')
    parser.add_argument('-seq_file', '--seq_file', type=str, help='path to the sequnce (.fasta format)',required=True)
    parser.add_argument('-save_path', '--save_path', type=str, help='path to the save directory',required=True)

    args = parser.parse_args()

    seq_file = args.seq_file
    final_msa_path = args.save_path

    search_msa(final_msa_path, seq_file)
