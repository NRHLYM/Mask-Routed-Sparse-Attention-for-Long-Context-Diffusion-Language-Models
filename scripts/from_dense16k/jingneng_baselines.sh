#!/usr/bin/env bash
# Jingneng baseline roots. Source after env.sh.
# Do not mix these PYTHONPATHs in one process.
#
# HiLS  (split-old / w256, currently training):
#   PYTHONPATH=$FULLTEACHER_ROOT
#   configs: $NSA_ROOT/configs/from_dense16k/hils-t4-jingneng-*-w256.json
#   output:  /Data/xiongjing/outputs/hils-t4-i4-hardroute-lora-qcal-lmk-klqcal-cedetach-w256
#
# NSA / SWA / DSA training code:
#   PYTHONPATH=$HILS_FT_ROOT  (/Data/xiongjing/src/hils-fullteacher-20260917)
#   Never default this to FULLTEACHER_ROOT: that tree rejects attention_mode=nsa/swa.
#   configs: $NSA_ROOT/configs/from_dense16k/*-jingneng.json  (Jingneng paths only)
# SWA noskipinert retrain/eval:
#   config:  $NSA_ROOT/configs/from_dense16k/swa-yarn8-16k-w1280-noskipinert-jingneng.json
#   output:  /Data/xiongjing/outputs/swa-yarn8-16k-w1280-noskipinert-fromdense
# Mixed sparse (21 SWA @ radius 1280 + 7 sparse), dual 4-GPU jobs:
#   machine1 0,1  HiLS  start_jingneng_lora_qcal_lmk_klqcal_cedetach_w256_swa1280.sh
#   machine1 2,3  NSA   start_jingneng_nsa_i4_w128_c64s64_swa1280.sh
#   machine2 0,1  DSA   start_jingneng_dsa_i4_k2560_swa1280.sh
#   machine2 2,3  SWA   start_jingneng_swa_w1280_noskipinert.sh
# Route CE ablations (each setting uses 4 GPUs; queue the second arm):
#   serverA  queue_jingneng_hils_route_s1_s3.sh   (S1 live fusion+teacher, then S3 mean-V STE)
#   serverB  queue_jingneng_hils_route_s2_s4.sh   (S2 no teacher, then S4 all-chunk ST)
#   launcher: SETTING=s1|s2|s3|s4 start_jingneng_hils_route_ablation.sh
#   PYTHONPATH=$FULLTEACHER_ROOT  (needs allchunk_gumbel.py)
# LongBench 2-gpu (wait for step-500, cwd $NSA_ROOT after env.sh):
#   HiLS  start_jingneng_longbench_hils_w256_swa1280.sh
#   NSA   start_jingneng_longbench_nsa_w128_swa1280.sh
#   DSA   start_jingneng_longbench_dsa_k2560_swa1280.sh
#   SWA   start_jingneng_longbench_swa_w1280.sh
# Official RULER probes (CPU prepare, then 2-gpu eval, YaRN L/2048 at 16k/32k/64k):
#   start_jingneng_ruler_prepare.sh
#   start_jingneng_ruler_2gpu.sh {hils|nsa|dsa|swa|dense}
#   tasks: hils_sn / hils_mkmq / hils_vt  (training 0/1/2, not stock 13-task YAML)
#   data:  /Data/xiongjing/data/ruler-probes
#   local_window = sparse branch; swa_local_window = 1280 for the 21 layers.
# Old tilde SWA (dropping inert) stays at .../swa-yarn8-16k-w1280-fromdense
# Fair eval: FA-SWA lives in FULLTEACHER_ROOT. Do not point NSA/DSA/SWA at the old tree.
HILS_FT_ROOT="${HILS_FT_ROOT:-/Data/xiongjing/src/hils-fullteacher-20260917}"
