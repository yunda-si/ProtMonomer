# -*- coding = utf-8 -*-
"""
Created on 2026/9/9
author: yunda_si@ucac.ac.cn
"""

import argparse
import io
import tarfile
import time
from pathlib import Path
import requests

HOST = "https://api.colabfold.com"


def read_single_fasta(filename):
    seq = []
    with open(filename) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if not line.startswith(">"):
                seq.append(line)

    sequence = "".join(seq)

    return sequence


def mmseqs_msa_search(
                      seq_file,
                      msa_file,
                      mode="env",
                      host=HOST
                      ):

    sequence = read_single_fasta(seq_file)
    query = f">101\n{sequence}\n"

    headers = {"User-Agent": "mmseqs_search.py/1.0"}
    print("Submitting MMseqs2 job")

    r = requests.post(
                  f"{host}/ticket/msa",
                      data={"q": query,"mode": mode},
                      headers=headers
                     )
    r.raise_for_status()
    job_id = r.json()["id"]
    print("Job ID:", job_id)

    while True:
        r = requests.get(f"{host}/ticket/{job_id}", headers=headers)
        r.raise_for_status()
        status = r.json()["status"]
        print("Status:", status)

        if status == "COMPLETE":
            break

        if status in ["ERROR", "UNKNOWN"]:
            raise RuntimeError(r.json())
        time.sleep(5)

    print("Downloading result...")
    r = requests.get(f"{host}/result/download/{job_id}", headers=headers)
    r.raise_for_status()

    tar_data = io.BytesIO(r.content)
    with tarfile.open(fileobj=tar_data, mode="r:gz") as tar:

        a3m_files = [x for x in tar.getnames() if x.endswith(".a3m")]
        if len(a3m_files) == 0:
            raise RuntimeError("No a3m file found")

        with open(msa_file, "w") as ff:
            for target in a3m_files:
                f = tar.extractfile(target)
                msa = f.read().decode()
                msa = "\n".join(
                                line if line.startswith(">") else "".join(c for c in line if not c.islower())
                                for line in msa.split("\n")
                                )
                ff.write(msa)
        print("MSA saved:",msa_file)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="ColabFold MMseqs2 MSA search")

    parser.add_argument(
          "--seq_file",
                        required=True,
                        help="input fasta"
                        )

    parser.add_argument(
          "--msa_file",
                        required=True,
                        help="output a3m"
                        )

    parser.add_argument(
          "--mode",
                        default="env",
                        choices=["env", "all"],
                        help="MMseqs2 database mode"
                        )

    args = parser.parse_args()
    mmseqs_msa_search(args.seq_file, args.msa_file, args.mode)