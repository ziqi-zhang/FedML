export PYTHONPATH=..:$PYTHONPATH

CUDA_VISIBLE_DEVICES=$1 \
python fedml_experiments/standalone/fedavg/main_fedavg.py \
--gpu 0 \
--dataset har_subject \
--data_dir data/UCIHAR \
--model cnn \
--partition_method homo  \
--client_num_in_total 10 \
--client_num_per_round 10 \
--comm_round 20 \
--epochs 10 \
--batch_size 64 \
--client_optimizer adam \
--lr 1e-4 \
--ci 0 \
--partition_alpha 1 \
--frequency_of_the_test 1 \
--run_tag baseline \