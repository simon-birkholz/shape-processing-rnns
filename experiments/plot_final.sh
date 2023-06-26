
Q="0.00625"
EXT="pdf"


# non-normalized dignostic stimuli

python plot_diagnostic_stimuli.py -f ../results/stimuli/2023-06-26-11-52_stimuli_conv-optim-normal.pt -o diagnostic_stimuli_conv_normal.$EXT --fdr $Q -n "Conv-Normal"

python plot_diagnostic_stimuli.py -f ../results/stimuli/2023-06-25-16-09_stimuli_conv-optim-wider.pt -o diagnostic_stimuli_conv_wider.$EXT --fdr $Q -n "Conv-Wider"

python plot_diagnostic_stimuli.py -f ../results/stimuli/2023-06-22-21-22_stimuli_rnn-skip-first-ts7.pt -o diagnostic_stimuli_rnn_ts7.$EXT --fdr $Q -n "RNN (T=7)"

python plot_diagnostic_stimuli.py -f ../results/stimuli/2023-06-23-00-03_stimuli_reciprocal-skip-first-ts7.pt -o diagnostic_stimuli_reciprocal_ts7.$EXT --fdr $Q -n "RGC (T=7)"

python plot_diagnostic_stimuli.py -f ../results/stimuli/2023-06-23-00-08_stimuli_gru-skip-first-ts7.pt -o diagnostic_stimuli_gru_ts7.$EXT --fdr $Q -n "GRU (T=7)"

python plot_diagnostic_stimuli.py -f ../results/stimuli/2023-06-24-01-01_stimuli_lstm-skip-first-ts7.pt -o diagnostic_stimuli_lstm_ts7.$EXT --fdr $Q -n "LSTM (T=7)"

python plot_diagnostic_stimuli.py -f ../results/stimuli/2023-06-25-15-48_stimuli_hgru-skip-first-ts7.pt -o diagnostic_stimuli_hgru_ts7.$EXT --fdr $Q -n "hGRU (T=7)"

python plot_diagnostic_stimuli.py -f ../results/stimuli/2023-06-25-15-52_stimuli_fgru-skip-first-ts7.pt -o diagnostic_stimuli_fgru_ts7.$EXT --fdr $Q -n "fGRU (T=7)"

#python plot_diagnostic_stimuli.py -f ../results/stimuli/2023-06-12-05-42_stimuli_fgru-ts3-non-normalized.pt -o diagnostic_stimuli_fgru_non_normalized.$EXT --fdr $Q -n "$\gamma$-Net (T=7)"


# rsa

python plot_rs_analysis.py -f ../results/rsa/2023-06-05-05-37_rsa_conv_fixed.pt -o rsa_conv_normal.$EXT --fdr $Q -n "Conv"

python plot_rs_analysis.py -f ../results/stimuli/2023-06-25-16-09_stimuli_conv-optim-wider.pt -o rsa_conv_wider.$EXT --fdr $Q -n "Conv-Wider"

python plot_rs_analysis.py -f ../results/rsa/2023-06-23-03-12_rsa_rnn-skip-first-ts7_fixed.pt -o rsa_rnn_ts7.$EXT --fdr $Q -n "RNN (T=7)"

python plot_rs_analysis.py -f ../results/rsa/2023-06-23-02-20_rsa_gru-skip-first-ts7_fixed.pt -o rsa_gru_ts7.$EXT --fdr $Q -n "GRU (T=7)"

python plot_rs_analysis.py -f ../results/rsa/2023-06-24-16-49_rsa_hgru-skip-first-ts7_fixed.pt -o rsa_hgru_ts7.$EXT --fdr $Q -n "hGRU (T=7)"

python plot_rs_analysis.py -f ../results/rsa/2023-06-25-01-54_rsa_lstm-skip-first-ts7_fixed.pt -o rsa_lstm_ts7.$EXT --fdr $Q -n "LSTM (T=7)"

python plot_rs_analysis.py -f ../results/rsa/2023-06-25-03-30_rsa_reciprocal-skip-first-ts7_fixed.pt -o rsa_rgc_ts7.$EXT --fdr $Q -n "RGC (T=7)"

python plot_rs_analysis.py -f ../results/rsa/2023-06-25-18-24_rsa_fgru-skip-first-ts7_fixed.pt -o rsa_fgru_ts7.$EXT --fdr $Q -n "fGRU (T=7)"
