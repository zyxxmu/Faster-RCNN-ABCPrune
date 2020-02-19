## Train

CUDA_VISIBLE_DEVICES=0,1,2 \
python trainval_net.py \
--dataset pascal_voc \
--net res101 \
--bs 1 \
--nw 4 \
--epochs 6 \
--lr 1e-3 \
--lr_decay_step 5 \
--cuda \
--mGPUs \
--use_tfb \
--save_dir /home/lmb/ABC-RCNN/result \
--honey 4 2 3 7 7 8 4 5 6 4 6 7 4 6 1 3 5 4 5 1 1 8 5 2 3 6 8 6 2 8 8 1 1 \
--honey_ckpt data/pretrained_model/resnet101.pt \


ln -s $VOCdevkit /media/disk2/zyc/voc2007/VOCdevkit

sudo chmod -R 777 /media/disk2/zyc/voc2007/VOCdevkit

python test_net.py \
--dataset pascal_voc \
--net res101 \
--ckpt /home/lmb/ABC-RCNN/result/res101/pascal_voc/faster_rcnn_1_6_10021.pth \
--cuda \
--honey 4 2 3 7 7 8 4 5 6 4 6 7 4 6 1 3 5 4 5 1 1 8 5 2 3 6 8 6 2 8 8 1 1 \