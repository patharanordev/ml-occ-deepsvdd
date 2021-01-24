mkdir -p ./log/cifar10_test && \
python src/main.py \
cifar10 \
cifar10_LeNet \
./log/cifar10_test \
./data \
--objective one-class \
--lr 0.0001 \
--n_epochs 150 \
--lr_milestone 50 \
--batch_size 200 \
--weight_decay 0.5e-6 \
--pretrain True \
--ae_lr 0.0001 \
--ae_n_epochs 350 \
--ae_lr_milestone 250 \
--ae_batch_size 200 \
--ae_weight_decay 0.5e-6 \
--normal_class 3