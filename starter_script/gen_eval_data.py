import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--encoding", default="pq", type=str, help="docid method atomic/pq/url")
parser.add_argument("--scale", default="top_300k", type=str, help="docid method atomic/pq/url")
args = parser.parse_args()

model = "t5"
encoding = args.encoding
max_seq_length = 128
sample_for_one_doc = 1
cur_data = "query_dev"
top_or_rand, scale = args.scale.split("_")
msmarco_or_nq = "msmarco"

def main():
    code_dir = "../"
    os.system(f"cd {code_dir}/gen_instance/ && python gen_{model}_eval_data.py \
        --max_seq_length {max_seq_length} \
        --pretrain_model_path {code_dir}/transformer_models/{model}-base \
        --data_path {code_dir}/dataset/{msmarco_or_nq}-data/{msmarco_or_nq}-docs-sents.{top_or_rand}.{scale}.json \
        --docid_path {code_dir}/dataset/encoded_docid/{model}_{encoding}_{top_or_rand}_{scale}.txt \
        --query_path {code_dir}/dataset/{msmarco_or_nq}-data/{msmarco_or_nq}-docdev-queries.tsv \
        --qrels_path {code_dir}/dataset/{msmarco_or_nq}-data/{msmarco_or_nq}-docdev-qrels.tsv \
        --output_path {code_dir}/dataset/test_data_{top_or_rand}_{scale}/query_dev.{model}_{max_seq_length}_{sample_for_one_doc}.{encoding}.{scale}.json \
        --current_data {cur_data}")

    print("write success")

if __name__ == '__main__':
    main()