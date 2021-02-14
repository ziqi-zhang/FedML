export PYTHONPATH=..:$PYTHONPATH

CUDA_VISIBLE_DEVICES=$1 \
python main.py \
--gpu_server_num 1 \
--gpu_num_per_server 1 \
--dataset har \
--data_dir ../../data/UCIHAR \
--model cnn \
--partition_method homo  \
--client_num_in_total 1 \
--client_num_per_round 1 \
--comm_round 20 \
--epochs 20 \
--batch_size 64 \
--client_optimizer adam \
--lr 1e-4 \
--ci 0 \
--partition_alpha 0.5 \
--frequency_of_test_acc_report 1 \
--frequency_of_train_acc_report 100 \
# --run_tag baseline \