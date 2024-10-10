# Load the SRR to GSM mapping from the other file into a dictionary
srr_to_gsm_mapping = {}

with open('sorted-SRR-GSM.txt') as f:
    for line in f:
        srr_tag, gsm_tag = line.strip().split('\t')
        srr_to_gsm_mapping[srr_tag] = gsm_tag
    import pdb;pdb.set_trace()

with open('descriptions.txt') as f2, open('SRR-labels.txt', 'w') as new_file:
   for line in f2:
        columns = line.strip().split('\t')
        # Extract the GSM tag from the first column
        gsm_tag = columns[0]

        # Look up the corresponding SRR tagsorted-SRR-GSM.txt using the GSM tag
        srr_tag = srr_to_gsm_mapping.get(gsm_tag, '')

        # Write the SRR tag followed by a tab as the new first column
        new_file.write(f'{srr_tag}\t')

        # Write the rest of the columns from the current line
        new_file.write(columns[1])
        #new_file.write('\t'.join(columns[1]))

        # Write a new line character to separate the rows
        new_file.write('\n')


