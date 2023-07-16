
Q="0.0125"
EXT="pdf"

# rsa

python timeseries_rs_analysis.py -f ../results/rsa/2023-06-25-15-34_rsa_rnn-skip-first-ts7-ro1_fixed.pt \
	-f ../results/rsa/2023-06-25-19-38_rsa_rnn-skip-first-ts7-ro2_fixed.pt \
	-f ../results/rsa/2023-06-25-20-35_rsa_rnn-skip-first-ts7-ro3_fixed.pt \
	-f ../results/rsa/2023-06-25-21-30_rsa_rnn-skip-first-ts7-ro4_fixed.pt \
	-f ../results/rsa/2023-06-25-22-27_rsa_rnn-skip-first-ts7-ro5_fixed.pt \
	-f ../results/rsa/2023-06-25-23-23_rsa_rnn-skip-first-ts7-ro6_fixed.pt \
	-f ../results/rsa/2023-06-26-00-23_rsa_rnn-skip-first-ts7-ro7_fixed.pt \
	-o rsa_rnn_ts1-7.$EXT --fdr $Q -n "RNN (T=7)"

python timeseries_rs_analysis.py -f ../results/rsa/2023-06-27-04-42_rsa_hgru-skip-first-ts7-ro1_fixed.pt \
	-f ../results/rsa/2023-06-27-05-32_rsa_hgru-skip-first-ts7-ro2_fixed.pt \
	-f ../results/rsa/2023-06-27-06-22_rsa_hgru-skip-first-ts7-ro3_fixed.pt \
	-f ../results/rsa/2023-06-27-07-12_rsa_hgru-skip-first-ts7-ro4_fixed.pt \
	-f ../results/rsa/2023-06-27-08-02_rsa_hgru-skip-first-ts7-ro5_fixed.pt \
	-f ../results/rsa/2023-06-27-08-52_rsa_hgru-skip-first-ts7-ro6_fixed.pt \
	-f ../results/rsa/2023-06-27-09-42_rsa_rnn-skip-first-ts7-ro7_fixed.pt \
	-o rsa_hgru_ts1-7.$EXT --fdr $Q -n "hGRU (T=7)"
	
	python timeseries_rs_analysis.py -f ../results/rsa/2023-07-07-03-16_rsa_fgru-skip-first-ts7-ro1_fixed.pt \
	-f ../results/rsa/2023-07-07-04-08_rsa_fgru-skip-first-ts7-ro2_fixed.pt \
	-f ../results/rsa/2023-07-07-05-00_rsa_fgru-skip-first-ts7-ro3_fixed.pt \
	-f ../results/rsa/2023-07-07-05-53_rsa_fgru-skip-first-ts7-ro4_fixed.pt \
	-f ../results/rsa/2023-07-07-06-45_rsa_fgru-skip-first-ts7-ro5_fixed.pt \
	-f ../results/rsa/2023-07-07-07-37_rsa_fgru-skip-first-ts7-ro6_fixed.pt \
	-f ../results/rsa/2023-07-07-08-29_rsa_fgru-skip-first-ts7-ro7_fixed.pt \
	-o rsa_fgru_ts1-7.$EXT --fdr $Q -n "fGRU (T=7)"
	
python timeseries_rs_analysis.py -f ../results/rsa/2023-07-05-18-02_rsa_reciprocal-skip-first-ts7-ro1_fixed.pt \
	-f ../results/rsa/2023-07-05-19-42_rsa_reciprocal-skip-first-ts7-ro2_fixed.pt \
	-f ../results/rsa/2023-07-05-21-15_rsa_reciprocal-skip-first-ts7-ro3_fixed.pt \
	-f ../results/rsa/2023-07-05-23-04_rsa_reciprocal-skip-first-ts7-ro4_fixed.pt \
	-f ../results/rsa/2023-07-06-00-54_rsa_reciprocal-skip-first-ts7-ro5_fixed.pt \
	-f ../results/rsa/2023-07-06-02-35_rsa_reciprocal-skip-first-ts7-ro6_fixed.pt \
	-f ../results/rsa/2023-07-06-04-09_rsa_reciprocal-skip-first-ts7-ro7_fixed.pt \
	-o rsa_rgc_ts1-7.$EXT --fdr $Q -n "RGC (T=7)"