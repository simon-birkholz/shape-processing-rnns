
Q="0.00625"
EXT="pdf"


# non-normalized

python plot_diagnostic_stimuli.py -f ../results/stimuli/2023-06-12-05-48_stimuli_conv-non-normalized.pt -o diagnostic_stimuli_conv_non_normalized.$EXT --fdr $Q -n "Conv"

python plot_diagnostic_stimuli.py -f ../results/stimuli/2023-06-22-21-22_stimuli_rnn-skip-first-ts7.pt -o diagnostic_stimuli_rnn_ts7.$EXT --fdr $Q -n "RNN (T=7)"

python plot_diagnostic_stimuli.py -f ../results/stimuli/2023-06-23-00-03_stimuli_reciprocal-skip-first-ts7.pt -o diagnostic_stimuli_reciprocal_ts7.$EXT --fdr $Q -n "RGC (T=7)"

python plot_diagnostic_stimuli.py -f ../results/stimuli/2023-06-23-00-08_stimuli_gru-skip-first-ts7.pt -o diagnostic_stimuli_gru_ts7.$EXT --fdr $Q -n "GRU (T=7)"

python plot_diagnostic_stimuli.py -f ../results/stimuli/2023-06-24-01-01_stimuli_lstm-skip-first-ts7.pt -o diagnostic_stimuli_lstm_ts7.$EXT --fdr $Q -n "LSTM (T=7)"

#python plot_diagnostic_stimuli.py -f ../results/stimuli/2023-06-12-05-39_stimuli_hgru-ts3-non-normalized.pt -o diagnostic_stimuli_hgru_non_normalized.$EXT --fdr $Q -n "hGRU (T=7)"

#python plot_diagnostic_stimuli.py -f ../results/stimuli/2023-06-12-05-42_stimuli_fgru-ts3-non-normalized.pt -o diagnostic_stimuli_fgru_non_normalized.$EXT --fdr $Q -n "fGRU (T=7)"

#python plot_diagnostic_stimuli.py -f ../results/stimuli/2023-06-12-05-42_stimuli_fgru-ts3-non-normalized.pt -o diagnostic_stimuli_fgru_non_normalized.$EXT --fdr $Q -n "$\gamma$-Net (T=7)"