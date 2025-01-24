python test.py --eval mobilenetv3s-01.25-05.38.46-acc@87.50-loss@0.6829-eval \
               --device cuda --num-workers 8 \
               --transform-file zoo/transform_instances/none.yaml \
               --model mobilenet_v3_s \
               --eval-acc \
               --checkpoint-path ckpts/mobilenetv3s_150e_8b_complex_sgd_1e-3_0.95_wd_1e-4_celoss_0.0/01.25-05.38.46-acc@87.50-loss@0.6829.pt \