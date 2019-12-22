python3 train.py \
  --data data/voc.data \
  --cfg cfg/yolov3-tiny-anchors.cfg \
  --weights weights/yolov3-tiny-ultralytics-pretrained.pt \
  --transfer \
  --rect \
  --device 0 \
  --batch-size 64 \
  --accumulate 1 \
  --epochs 68 \
  --img-weights \
  --adam \
  --prebias \
  --multi-scale \
#  --evolve \
