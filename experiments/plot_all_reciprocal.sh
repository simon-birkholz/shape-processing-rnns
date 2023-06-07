
Q="0.05"
EXT="png"
# plot diagnostic stimuli plots

python plot_diagnostic_stimuli.py -f ../results/stimuli/2023-06-06-04-54_stimuli_reciprocal-preconv-ts3-ro1.pt -o diagnostic_stimuli_reciprocal_ts3_ro1.$EXT --fdr $Q -n "3T-RGC (T=1)"
python plot_diagnostic_stimuli.py -f ../results/stimuli/2023-06-06-05-03_stimuli_reciprocal-preconv-ts3-ro2.pt -o diagnostic_stimuli_reciprocal_ts3_ro2.$EXT --fdr $Q -n "3T-RGC (T=2)"
python plot_diagnostic_stimuli.py -f ../results/stimuli/2023-06-05-16-25_stimuli_reciprocal-preconv-ts3-ro3.pt -o diagnostic_stimuli_reciprocal_ts3_ro3.$EXT --fdr $Q -n "3T-RGC (T=3)"
python plot_diagnostic_stimuli.py -f ../results/stimuli/2023-06-05-16-34_stimuli_reciprocal-preconv-ts3-ro5.pt -o diagnostic_stimuli_reciprocal_ts3_ro5.$EXT --fdr $Q -n "3T-RGC (T=5)"
python plot_diagnostic_stimuli.py -f ../results/stimuli/2023-06-06-04-47_stimuli_reciprocal-preconv-ts3-ro7.pt -o diagnostic_stimuli_reciprocal_ts3_ro7.$EXT --fdr $Q -n "3T-RGC (T=7)"

# plot rsa plots

python plot_rs_analysis.py -f ../results/rsa/2023-06-05-19-34_rsa_reciprocal-preconv-ts3-ro1-hidden_fixed.pt -o rsa_reciprocal-ts3-ro1-hidden_fixed.$EXT --fdr $Q -n "3T-RGC (T=1, Hidden State)"
python plot_rs_analysis.py -f ../results/rsa/2023-06-05-20-30_rsa_reciprocal-preconv-ts3-ro2-hidden_fixed.pt -o rsa_reciprocal-ts3-ro2-hidden_fixed.$EXT --fdr $Q -n "3T-RGC (T=2, Hidden State)"
python plot_rs_analysis.py -f ../results/rsa/2023-06-05-21-27_rsa_reciprocal-preconv-ts3-ro3-hidden_fixed.pt -o rsa_reciprocal-ts3-ro3-hidden_fixed.$EXT --fdr $Q -n "3T-RGC (T=3, Hidden State)"
python plot_rs_analysis.py -f ../results/rsa/2023-06-05-22-26_rsa_reciprocal-preconv-ts3-ro5-hidden_fixed.pt -o rsa_reciprocal-ts3-ro5-hidden_fixed.$EXT --fdr $Q -n "3T-RGC (T=5, Hidden State)"
python plot_rs_analysis.py -f ../results/rsa/2023-06-05-23-24_rsa_reciprocal-preconv-ts3-ro7-hidden_fixed.pt -o rsa_reciprocal-ts3-ro7-hidden_fixed.$EXT --fdr $Q -n "3T-RGC (T=7, Hidden State)"


python plot_rs_analysis.py -f ../results/rsa/2023-06-06-00-35_rsa_reciprocal-preconv-ts3-ro1-cell_fixed.pt -o rsa_reciprocal-ts3-ro1-cell_fixed.$EXT --fdr $Q -n "3T-RGC (T=1, Cell State)"
python plot_rs_analysis.py -f ../results/rsa/2023-06-06-01-34_rsa_reciprocal-preconv-ts3-ro2-cell_fixed.pt -o rsa_reciprocal-ts3-ro2-cell_fixed.$EXT --fdr $Q -n "3T-RGC (T=2, Cell State)"
python plot_rs_analysis.py -f ../results/rsa/2023-06-06-02-32_rsa_reciprocal-preconv-ts3-ro3-cell_fixed.pt -o rsa_reciprocal-ts3-ro3-cell_fixed.$EXT --fdr $Q -n "3T-RGC (T=3, Cell State)"
python plot_rs_analysis.py -f ../results/rsa/2023-06-06-03-30_rsa_reciprocal-preconv-ts3-ro5-cell_fixed.pt -o rsa_reciprocal-ts3-ro5-cell_fixed.$EXT --fdr $Q -n "3T-RGC (T=5, Cell State)"
python plot_rs_analysis.py -f ../results/rsa/2023-06-06-04-30_rsa_reciprocal-preconv-ts3-ro7-cell_fixed.pt -o rsa_reciprocal-ts3-ro7-cell_fixed.$EXT --fdr $Q -n "3T-RGC (T=7, Cell State)"