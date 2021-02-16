export PYTHONPATH=..:$PYTHONPATH

CUDA_VISIBLE_DEVICES=$1 \
python fedml_experiments/standalone/fedavg/main_fedavg.py \
--gpu 0 \
--dataset adult \
--data_dir data/UCIAdult \
--model lr \
--partition_method homo  \
--client_num_in_total 10 \
--client_num_per_round 10 \
--comm_round 20 \
--epochs 5 \
--batch_size 128 \
--client_optimizer sgd \
--lr 0.03 \
--ci 0 \
--frequency_of_the_test 1 \
--run_tag baseline \