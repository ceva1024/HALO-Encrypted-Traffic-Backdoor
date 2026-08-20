# Running the experiments

## 0. Environment

```bash
export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=8
export MPLCONFIGDIR=/tmp/matplotlib
export PYTHONUNBUFFERED=1
mkdir -p models logs
```

## 1. Train the clean classifier

```bash
python train/clean.py \
  --train_jsonl data/train.jsonl \
  --valid_jsonl data/valid.jsonl \
  --test_jsonl data/test.jsonl \
  --task app \
  --save_dir models/clean \
  --backbone lstm \
  --rnn_layers 3 \
  --d_model 128 \
  --rnn_hidden 192 \
  --max_len 128 \
  --batch_size 64 \
  --epochs 28 \
  --checkpoint_policy last \
  --lr 0.0055 \
  --weight_decay 3e-05 \
  --dropout 0.1 \
  --no_dt_bucket \
  --seed 42 \
  --num_workers 0 \
  --log_interval 0 \
  2>&1 | tee logs/clean.log
```

## 2. Train and evaluate HALO

```bash
python train/HALO.py \
  --train_jsonl data/train.jsonl \
  --valid_jsonl data/valid.jsonl \
  --test_jsonl data/test.jsonl \
  --task app \
  --poison_source_labels 7 \
  --target_class 1 \
  --save_dir models/halo \
  --clean_model_path models/clean/final_classifier_model.pt \
  --backbone lstm \
  --rnn_layers 3 \
  --d_model 128 \
  --rnn_hidden 192 \
  --gen_rnn_type lstm \
  --gen_hidden 128 \
  --gen_layers 2 \
  --max_len 128 \
  --batch_size 64 \
  --epochs 9 \
  --lr 0.0041 \
  --poison_rate 1.0 \
  --only_server \
  --lambda_poison 0.4 \
  --lambda_pad 0.0001 \
  --lambda_align 20 \
  --weight_decay 3e-05 \
  --dropout 0.1 \
  --num_target_prototypes 8 \
  --max_proto_samples 2000 \
  --proto_soft_tau 0.08 \
  --proto_refresh_interval 6 \
  --proto_online_momentum 0.0001 \
  --proto_radius_percentile 80 \
  --lambda_repulse 0.01 \
  --repulse_delta 0.06 \
  --no_dt_bucket \
  --save_every_epoch \
  --seed 42 \
  --num_workers 0 \
  --log_interval 0 \
  2>&1 | tee logs/halo.log
```

Generate the triggered file referenced by the defense command below:

```bash
python experiments/evaluate_halo_joint_epochs.py \
  --trial_dir models/halo \
  --checkpoint models/halo/joint_epoch_009.pt \
  --out_dir models/halo/test_tuning_eval \
  --eval_name test \
  2>&1 | tee logs/halo_epoch009_eval.log
```

## 3. Train the attack baselines

### 3.1 BadNets

```bash
python train/badnets.py \
  --train_jsonl data/train.jsonl \
  --valid_jsonl data/valid.jsonl \
  --test_jsonl data/test.jsonl \
  --task app \
  --poison_source_labels 7 \
  --target_class 1 \
  --save_dir models/badnets \
  --clean_model_path models/clean/final_classifier_model.pt \
  --backbone lstm \
  --rnn_layers 3 \
  --d_model 128 \
  --rnn_hidden 192 \
  --max_len 128 \
  --batch_size 64 \
  --epochs 5 \
  --lr 0.0041 \
  --poison_rate 1.0 \
  --only_server \
  --lambda_poison 1.0 \
  --bd_insert_lens 1081,905,792 \
  --bd_after_kth_server 1 \
  --bd_dt_mode const \
  --bd_dt_const 1.0 \
  --weight_decay 3e-05 \
  --dropout 0.1 \
  --no_dt_bucket \
  --save_every_epoch \
  --seed 42 \
  --num_workers 0 \
  --log_interval 0 \
  2>&1 | tee logs/badnets.log
```

### 3.2 TrojanFlow

```bash
python train/trojanflow.py \
  --train_jsonl data/train.jsonl \
  --valid_jsonl data/valid.jsonl \
  --test_jsonl data/test.jsonl \
  --task app \
  --poison_source_labels 7 \
  --target_class 1 \
  --save_dir models/trojanflow \
  --clean_model_path models/clean/final_classifier_model.pt \
  --backbone lstm \
  --rnn_layers 3 \
  --d_model 128 \
  --rnn_hidden 192 \
  --gen_rnn_type lstm \
  --gen_hidden 128 \
  --gen_layers 2 \
  --max_len 128 \
  --batch_size 64 \
  --epochs 8 \
  --lr 0.001 \
  --poison_rate 1.0 \
  --only_server \
  --lambda_poison 1.0 \
  --weight_decay 3e-05 \
  --dropout 0.1 \
  --no_dt_bucket \
  --save_every_epoch \
  --seed 42 \
  --num_workers 0 \
  --log_interval 0 \
  2>&1 | tee logs/trojanflow.log
```

### 3.3 UAP

```bash
python train/uap.py \
  --train_jsonl data/train.jsonl \
  --valid_jsonl data/valid.jsonl \
  --test_jsonl data/test.jsonl \
  --task app \
  --poison_source_labels 7 \
  --target_class 1 \
  --save_dir models/uap \
  --clean_model_path models/clean/final_classifier_model.pt \
  --backbone lstm \
  --rnn_layers 3 \
  --d_model 128 \
  --rnn_hidden 192 \
  --max_len 128 \
  --batch_size 64 \
  --epochs 5 \
  --lr 0.0041 \
  --poison_rate 1.0 \
  --only_server \
  --lambda_poison 1.0 \
  --uap_max_bytes 4096 \
  --uap_init_bytes 128 \
  --uap_integer \
  --uap_lr 0.0041 \
  --uap_opt_epochs 2 \
  --weight_decay 3e-05 \
  --dropout 0.1 \
  --no_dt_bucket \
  --save_every_epoch \
  --seed 42 \
  --num_workers 0 \
  --log_interval 0 \
  2>&1 | tee logs/uap.log
```

## 4. Run the three backdoor defenses

Define this shell function once. It runs filtered SCAn, Beatrix, and TED for
one checkpoint/triggered-file pair.

```bash
run_backdoor_defenses() {
  ATTACK_DIR="$1"
  CKPT="$2"
  TRIGGERED="$3"
  OUT_ROOT="$4"

  python detect/scan.py \
    --ckpt "${CKPT}" \
    --clean_jsonl "${ATTACK_DIR}/train_subset_per_class.jsonl" \
    --mix_base_jsonl "${ATTACK_DIR}/train_subset_per_class.jsonl" \
    --inspect_jsonl "${TRIGGERED}" \
    --inspect_use_triggered_lengths \
    --force_label_id 1 \
    --out_dir "${OUT_ROOT}/scan_detect_filtered" \
    --max_len 128 \
    --batch_size 128 \
    --scan_min_samples 20 \
    --threshold_z 2 \
    --mc_samples 100 \
    --lrt_dim 32 \
    --seed 42 \
    --filter_no_inspect \
    --min_inspect_count 1 \
    --min_pi 0.03 \
    --bic_lambda 1.0

  python detect/beatrix.py \
    --cls_ckpt "${CKPT}" \
    --clean_subset_jsonl "${ATTACK_DIR}/train_subset_per_class.jsonl" \
    --triggered_jsonl "${TRIGGERED}" \
    --output_dir "${OUT_ROOT}/beatrix_detect" \
    --max_len 128 \
    --batch_size 128 \
    --max_order 3

  python detect/ted.py \
    --cls_ckpt "${CKPT}" \
    --clean_subset_jsonl "${ATTACK_DIR}/train_subset_per_class.jsonl" \
    --triggered_jsonl "${TRIGGERED}" \
    --output_dir "${OUT_ROOT}/ted_detect" \
    --max_len 128 \
    --batch_size 128 \
    --threshold_percentile 95
}
```

Run the function for HALO and all three baselines:

```bash
run_backdoor_defenses \
  models/halo \
  models/halo/joint_epoch_009.pt \
  models/halo/test_tuning_eval/joint_epoch_009/test_triggered_poisoned.jsonl \
  models/halo/detectors

run_backdoor_defenses \
  models/badnets \
  models/badnets/joint_epoch_005.pt \
  models/badnets/test_triggered_poisoned.jsonl \
  models/badnets/detectors

run_backdoor_defenses \
  models/trojanflow \
  models/trojanflow/joint_epoch_008.pt \
  models/trojanflow/test_triggered_poisoned.jsonl \
  models/trojanflow/detectors

run_backdoor_defenses \
  models/uap \
  models/uap/joint_epoch_005.pt \
  models/uap/test_triggered_poisoned.jsonl \
  models/uap/detectors
```
