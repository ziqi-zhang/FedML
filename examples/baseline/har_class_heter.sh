export PYTHONPATH=..:$PYTHONPATH

CUDA_VISIBLE_DEVICES=$1 \
python fedml_experiments/standalone/fedavg/main_fedavg.py \
--gpu 0 \
--dataset har \
--data_dir data/UCIHAR \
--model cnn \
--partition_method p-hetero  \
--client_num_in_total 6 \
--client_num_per_round 6 \
--comm_round 20 \
--epochs 10 \
--batch_size 64 \
--client_optimizer adam \
--lr 1e-4 \
--ci 0 \
--partition_alpha 0.5 \
--frequency_of_the_test 1 \