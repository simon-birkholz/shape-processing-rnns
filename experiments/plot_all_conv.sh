
Q="0.05"
EXT="png"
# plot diagnostic stimuli plots

python plot_diagnostic_stimuli.py -f ../results/stimuli/2023-06-05-04-35_stimuli_conv.pt -o diagnostic_stimuli_conv.$EXT --fdr $Q -n "Conv"

# plot rsa plots

python plot_rs_analysis.py -f ../results/rsa/2023-06-05-05-37_rsa_conv_fixed.pt -o rsa_conv_fixed.$EXT --fdr $Q -n "Conv"
