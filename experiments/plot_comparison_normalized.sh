
Q="0.00625"
EXT="png"


# plot diagnostic stimuli plots (normalized)

python plot_diagnostic_stimuli.py -f ../results/stimuli/2023-06-05-04-35_stimuli_conv.pt -o diagnostic_stimuli_conv_normalized.$EXT --fdr $Q -n "Conv"

python plot_diagnostic_stimuli.py -f ../results/stimuli/2023-06-05-04-14_stimuli_rnn-ts3-ro3.pt -o diagnostic_stimuli_rnn_normalized.$EXT --fdr $Q -n "RNN (T=3)"

python plot_diagnostic_stimuli.py -f ../results/stimuli/2023-06-05-16-25_stimuli_reciprocal-preconv-ts3-ro3.pt -o diagnostic_stimuli_reciprocal_normalized.$EXT --fdr $Q -n "RGC (T=3)"

python plot_diagnostic_stimuli.py -f ../results/stimuli/2023-06-12-04-56_stimuli_gru-ts3-normalized.pt -o diagnostic_stimuli_gru_normalized.$EXT --fdr $Q -n "GRU (T=3)"

python plot_diagnostic_stimuli.py -f ../results/stimuli/2023-06-12-05-20_stimuli_hgru-ts3-normalized.pt -o diagnostic_stimuli_hgru_normalized.$EXT --fdr $Q -n "hGRU (T=3)"

python plot_diagnostic_stimuli.py -f ../results/stimuli/2023-06-12-05-25_stimuli_fgru-ts3-normalized.pt -o diagnostic_stimuli_fgru_normalized.$EXT --fdr $Q -n "fGRU (T=3)"

# non-normalized

python plot_diagnostic_stimuli.py -f ../results/stimuli/2023-06-12-05-48_stimuli_conv-non-normalized.pt -o diagnostic_stimuli_conv_non_normalized.$EXT --fdr $Q -n "Conv"

python plot_diagnostic_stimuli.py -f ../results/stimuli/2023-06-07-05-18_stimuli_rnn-ts3-non-normalized.pt -o diagnostic_stimuli_rnn_non_normalized.$EXT --fdr $Q -n "RNN (T=3)"

#python plot_diagnostic_stimuli.py -f ../results/stimuli/2023-06-05-16-25_stimuli_reciprocal-preconv-ts3-ro3.pt -o diagnostic_stimuli_reciprocal_normalized.$EXT --fdr $Q -n "RGC (T=3)"

python plot_diagnostic_stimuli.py -f ../results/stimuli/2023-06-12-05-36_stimuli_gru-ts3-non-normalized.pt -o diagnostic_stimuli_gru_non_normalized.$EXT --fdr $Q -n "GRU (T=3)"

python plot_diagnostic_stimuli.py -f ../results/stimuli/2023-06-12-05-39_stimuli_hgru-ts3-non-normalized.pt -o diagnostic_stimuli_hgru_non_normalized.$EXT --fdr $Q -n "hGRU (T=3)"

python plot_diagnostic_stimuli.py -f ../results/stimuli/2023-06-12-05-42_stimuli_fgru-ts3-non-normalized.pt -o diagnostic_stimuli_fgru_non_normalized.$EXT --fdr $Q -n "fGRU (T=3)"