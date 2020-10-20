CUDA_VISIBLE_DEVICES=0,1,2,3,4,9,5,6,7,8  ./scripts/dist_run_single.sh 5 vcr/train_end2end.py cfgs/vcr/base_q2a_4x16G_fp32.yaml .tmp
