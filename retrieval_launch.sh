FILE_PATH=PATH/TO/COURPUS/AND/INDEX/FILES
INDEX_FILE=$FILE_PATH/wiki-18-e5-index/e5_Flat.index
CORPUS_FILE=$FILE_PATH/wiki-18-corpus/wiki-18.jsonl
RETRIEVER_NAME=e5
RETRIEVER_PATH=PATH/TO/RETRIEVER/e5-base-v2
PORT=8000

python reason_search/search/retrieval_server.py --index_path $INDEX_FILE \
                                            --corpus_path $CORPUS_FILE \
                                            --topk 3 \
                                            --retriever_name $RETRIEVER_NAME \
                                            --retriever_model $RETRIEVER_PATH \
                                            --faiss_gpu \
                                            --port $PORT
                                            