#!/usr/bin/env python3
# -*- coding:utf-8 -*-

import gzip
import sys

from collections import defaultdict

def main(gencode_fasta, salmon_quant):
   # Get transcript-genes associations.
   gene = dict()
   with gzip.open(gencode_fasta) as f:
      for line in f:
         line = line.decode("ascii")
         if line[0] != ">": continue
         ENST, ENSG, _ = line[1:].split("|", maxsplit=2)
         gene[ENST] = ENSG
   # Count expression per gene.
   counts = defaultdict(float)
   with open(salmon_quant) as f:
      _ = next(f)
      for line in f:
         tx, _, _, _, nb = line.split()
         counts[gene.get(tx, tx)] += float(nb)
   # Spit it out.
   for gene in sorted(counts):
      print(gene, counts[gene])


if __name__ == "__main__":
   main(sys.argv[1], sys.argv[2])
