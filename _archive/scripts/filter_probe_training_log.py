input_log_file = "/projectnb/mcnet/jbrin/lang-probing/run/word_probes_for_in_out.out"
output_log_file = "/projectnb/mcnet/jbrin/lang-probing/run/word_probes_for_in_out_filtered.out"

with open(input_log_file, 'r') as f:
    lines = f.readlines()

with open(output_log_file, 'w') as f:
    for line in lines:
        if line.startswith("INFO"):
            f.write(line)