
Q="0.05"
EXT="png"
# plot diagnostic stimuli plots

python plot_diagnostic_stimuli.py -f ../results/stimuli/2023-05-21-19-50_stimuli_rnn_ts3.pt -o diagnostic_stimuli_rnn_ts3.$EXT --fdr $Q -n "RNN (T=3)"
python plot_diagnostic_stimuli.py -f ../results/stimuli/2023-05-23-21-41_stimuli_rnn-ts5.pt -o diagnostic_stimuli_rnn_ts5.$EXT --fdr $Q -n "RNN (T=5)"

# plot rsa plots

python plot_rs_analysis.py -f ../results/rsa/2023-05-24-01-12_rsa_conv_fixed.pt -o rsa_conv_fixed.$EXT --fdr $Q -n "Conv"

python plot_rs_analysis.py -f ../results/rsa/2023-05-15-21-23_rsa_rnn-ts3_fixed.pt -o rsa_rnn-ts3_fixed.$EXT --fdr $Q -n "RNN (T=3)"
python plot_rs_analysis.py -f ../results/rsa/2023-05-23-23-31_rsa_rnn-ts5_fixed.pt -o rsa_rnn-ts5_fixed.$EXT --fdr $Q -n "RNN (T=5)"