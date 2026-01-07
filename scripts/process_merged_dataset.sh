DATA_DIR='PATH/TO/DATA/FlashRAG_datasets'
WORK_DIR='WORKING/DIR/PATH'

LOCAL_DIR=$WORK_DIR/data/nq_hotpotqa_train


DATA=nq,hotpotqa
PYTHONUNBUFFERED=1 python3 -m scripts.data_process.qa_search_train_merge --local_dir $LOCAL_DIR --data_sources $DATA --data_dir $DATA_DIR --template_type base
PYTHONUNBUFFERED=1 python3 -m scripts.data_process.qa_search_train_merge --local_dir $LOCAL_DIR --data_sources $DATA --data_dir $DATA_DIR --template_type verifier


DATA=nq,triviaqa,popqa,hotpotqa,2wikimultihopqa,musique,bamboogle
PYTHONUNBUFFERED=1 python3 -m scripts.data_process.qa_search_test_merge --local_dir $LOCAL_DIR --data_sources $DATA --data_dir $DATA_DIR --template_type base
PYTHONUNBUFFERED=1 python3 -m scripts.data_process.qa_search_test_merge --local_dir $LOCAL_DIR --data_sources $DATA --data_dir $DATA_DIR --template_type verifier