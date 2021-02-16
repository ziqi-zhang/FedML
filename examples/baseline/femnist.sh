export PYTHONPATH=..:$PYTHONPATH


python fedml_experiments/standalone/fedavg/main_fedavg.py \
--gpu $1 \
--dataset femnist \
--data_dir data/FederatedEMNIST/datasets \
--model cnn \
--partition_method homo  \
--client_num_in_total 10 \
--client_num_per_round 10 \
--comm_round 20 \
--epochs 1 \
--batch_size 100 \
--client_optimizer sgd \
--lr 0.03 \
--ci 0 \
--frequency_of_the_test 1 \
--partition_alpha 0.5 \
--run_tag baseline \