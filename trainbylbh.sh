/home/hantang/anaconda3/envs/GHAW/bin/torchrun \
  --nproc_per_node=2 \
  /home/hantang/usr/lbh/GHAW/tools/train.py \
  --config /home/hantang/usr/lbh/GHAW/projects/rtmpose/rtmpose/wholebody_2d_keypoint/rtmpose-x_8xb32-270e_coco-wholebody-384x288.py \
  --launcher pytorch