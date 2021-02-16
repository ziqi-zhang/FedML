export PYTHONPATH=..:$PYTHONPATH

CUDA_VISIBLE_DEVICES=$1 \
python fedml_experiments/standalone/fedavg/main_fedavg.py \
--gpu 0 \
--dataset cifar10 \
--data_dir data/cifar10 \
--model vgg11 \
--partition_method hetero  \
--client_num_in_total 10 \
--client_num_per_round 10 \
--comm_round 20 \
--epochs 1 \
--batch_size 100 \
--client_optimizer sgd \
--lr 0.03 \
--ci 0 \
--frequency_of_the_test 1 \