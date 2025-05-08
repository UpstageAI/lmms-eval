input_dir="/app/docfm/datasets/benchmark/key_information_extraction/v3.1/"
output_dir="./v3.1_dp/" #"/app/docfm/datasets/benchmark/key_information_extraction/v3.1_dp/"

CUDA_VISIBLE_DEVICES=4 python KIE_bench_dp.py --input_dir $input_dir --output_dir $output_dir --num_chunk 1 --start_chunk_index 0