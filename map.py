srr_mapping = {}

with open('modified_labels.txt') as f:
   for line in f:
      srr_tag, label = line.strip().split('\t')
      srr_mapping[srr_tag] = label

with open('transposed_expr.tsv') as f2, open('modified_matrix.txt', 'w') as new_file:
   next(f2)
   for line in f2:
      columns = line.split(' ')
      SRR_tag = line.split(' ')[0]
      label = srr_mapping.get(SRR_tag, '') 

      # Write the SRR tag followed by a space as the new first column
      new_file.write(f'{label} ')
      # Write the rest of the columns from the current line
      new_file.write(' '.join(str(x) for x in line.split(' ')[1:]))





