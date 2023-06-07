
Q="0.05"
EXT="png"
# plot diagnostic stimuli plots

python plot_diagnostic_stimuli.py -f ../results/stimuli/2023-06-05-04-21_stimuli_rnn-ts3-ro1.pt -o diagnostic_stimuli_rnn_ts3_ro1.$EXT --fdr $Q -n "3T-RNN (T=1)"
python plot_diagnostic_stimuli.py -f ../results/stimuli/2023-06-05-04-17_stimuli_rnn-ts3-ro2.pt -o diagnostic_stimuli_rnn_ts3_ro2.$EXT --fdr $Q -n "3T-RNN (T=2)"
python plot_diagnostic_stimuli.py -f ../results/stimuli/2023-06-05-04-14_stimuli_rnn-ts3-ro3.pt -o diagnostic_stimuli_rnn_ts3_ro3.$EXT --fdr $Q -n "3T-RNN (T=3)"
python plot_diagnostic_stimuli.py -f ../results/stimuli/2023-06-05-04-25_stimuli_rnn-ts3-ro5.pt -o diagnostic_stimuli_rnn_ts3_ro5.$EXT --fdr $Q -n "3T-RNN (T=5)"
python plot_diagnostic_stimuli.py -f ../results/stimuli/2023-06-05-04-29_stimuli_rnn-ts3-ro7.pt -o diagnostic_stimuli_rnn_ts3_ro7.$EXT --fdr $Q -n "3T-RNN (T=7)"

# plot rsa plots

python plot_rs_analysis.py -f ../results/rsa/2023-06-05-06-36_rsa_rnn-ts3-ro1_fixed.pt -o rsa_rnn-ts3-ro1_fixed.$EXT --fdr $Q -n "3T-RNN (T=1)"
python plot_rs_analysis.py -f ../results/rsa/2023-06-05-07-32_rsa_rnn-ts3-ro2_fixed.pt -o rsa_rnn-ts3-ro2_fixed.$EXT --fdr $Q -n "3T-RNN (T=2)"
python plot_rs_analysis.py -f ../results/rsa/2023-06-05-08-29_rsa_rnn-ts3-ro3_fixed.pt -o rsa_rnn-ts3-ro3_fixed.$EXT --fdr $Q -n "3T-RNN (T=3)"
python plot_rs_analysis.py -f ../results/rsa/2023-06-05-09-25_rsa_rnn-ts3-ro5_fixed.pt -o rsa_rnn-ts3-ro5_fixed.$EXT --fdr $Q -n "3T-RNN (T=5)"
python plot_rs_analysis.py -f ../results/rsa/2023-06-05-10-22_rsa_rnn-ts3-ro7_fixed.pt -o rsa_rnn-ts3-ro7_fixed.$EXT --fdr $Q -n "3T-RNN (T=7)"
