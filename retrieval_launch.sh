export NO_PROXY="127.0.0.1,localhost"

file_path=/mnt/cfs_algo_bj/models/experiments/xucan05/data
index_file=$file_path/wiki-18-e5-index/e5_Flat.index
corpus_file=$file_path/wiki-18-corpus/wiki-18.jsonl
retriever_name=e5
retriever_path=/mnt/cfs_algo_bj/models/opensource_model/e5-base-v2

python reason_search/search/retrieval_server.py --index_path $index_file \
                                            --corpus_path $corpus_file \
                                            --topk 3 \
                                            --retriever_name $retriever_name \
                                            --retriever_model $retriever_path \
                                            --faiss_gpu
                                            
export https_proxy=http://agent.baidu.com:8891
export http_proxy=http://agent.baidu.com:8891