
Q="0.05"
# plot diagnostic stimuli plots

python plot_diagnostic_stimuli.py -f ../results/stimuli/2023-05-21-19-50_rnn_ts3.pt -o diagnostic_stimuli_rnn_ts3.pdf --fdr $Q

# plot rsa plots

python plot_rs_analysis.py -f ../results/rsa/2023-05-15-21-23_rsa_rnn-ts3_fixed.pt -o rsa_rnn-ts3_fixed.pdf --fdr $Q