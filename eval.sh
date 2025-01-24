python test.py --device cpu \
               --transform-file zoo/transform_instances/none.yaml \
               --model mobilenet_v3_s \
               --checkpoint-path ckpts/01.24-06:41:18/01.24-06:55:17-Accuracy@75.5208-Loss@0.78.pt \
               --eval-acc \