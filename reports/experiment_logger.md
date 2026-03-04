## Best performing 
Model: PtVId (bert-large) (liaad/PtVId)

  DSL-TL:
  F1 (binary): 84.9722%
  F1 (macro):  79.7502%
  Accuracy:    81.0968%
              precision    recall  f1-score   support

       PT-PT       0.65      0.88      0.75       269
       PT-BR       0.93      0.78      0.85       588

    accuracy                           0.81       857
   macro avg       0.79      0.83      0.80       857
weighted avg       0.84      0.81      0.82       857

  FRMT:
  F1 (binary): 77.2532%
  F1 (macro):  76.6592%
  Accuracy:    76.6743%
              precision    recall  f1-score   support

       PT-PT       0.78      0.74      0.76      2614
       PT-BR       0.75      0.79      0.77      2612

    accuracy                           0.77      5226
   macro avg       0.77      0.77      0.77      5226
weighted avg       0.77      0.77      0.77      5226

Best performing F1 for PT-PT
    * DSL-TL: 75%
    * FRMT: 76%

## bastao/PeroVaz_PT-BR_Classifier

DSL-TL:
              precision    recall  f1-score   support

       PT-PT       0.53      0.81      0.64       269
       PT-BR       0.89      0.67      0.77       588

    accuracy                           0.72       857
   macro avg       0.71      0.74      0.71       857
weighted avg       0.78      0.72      0.73       857

FRMT:
              precision    recall  f1-score   support

       PT-PT       0.67      0.60      0.63      2614
       PT-BR       0.64      0.70      0.67      2612

    accuracy                           0.65      5226
   macro avg       0.65      0.65      0.65      5226
weighted avg       0.65      0.65      0.65      5226


DSL-TL PT-PT F1: 64.4%
FRMT PT-PT F1:   63.0%

## Experiment: journalistic_baseline
- date: 2026-03-04 10:26 UTC
- script: train_journalistic.py
- In-domain PT-PT F1: 97.4%
- DSL-TL PT-PT F1: 64.3%
- FRMT PT-PT F1: 46.1%

## Experiment: journalistic_baseline
- date: 2026-03-04 10:27 UTC
- script: train_journalistic.py
- In-domain PT-PT F1: 97.4%
- DSL-TL PT-PT F1: 64.9%
- FRMT PT-PT F1: 47.1%
- dataset: journalistic
- name: journalistic_baseline
- max_per_class: 100000
- fasttext_lr: 0.8
- fasttext_epoch: 3
- fasttext_wordNgrams: 1
- fasttext_minn: 2
- fasttext_maxn: 5
- fasttext_dim: 256
- fasttext_bucket: 1000000
- fasttext_minCount: 500
- fasttext_loss: softmax
- fasttext_thread: 48

## Experiment: veracruz_baseline
- date: 2026-03-04 10:31 UTC
- script: train_veracruz.py
- In-domain PT-PT F1: 86.5%
- DSL-TL PT-PT F1: 68.8%
- FRMT PT-PT F1: 73.2%
- name: veracruz_baseline
- fasttext_lr: 0.8
- fasttext_epoch: 3
- fasttext_wordNgrams: 1
- fasttext_minn: 2
- fasttext_maxn: 5
- fasttext_dim: 256
- fasttext_bucket: 1000000
- fasttext_minCount: 500
- fasttext_loss: softmax
- fasttext_thread: 48

## Experiment: veracruz_large
- date: 2026-03-04 10:44 UTC
- script: train_veracruz.py
- In-domain PT-PT F1: 87.5%
- DSL-TL PT-PT F1: 70.5%
- FRMT PT-PT F1: 73.8%
- name: veracruz_large
- fasttext_lr: 0.8
- fasttext_epoch: 3
- fasttext_wordNgrams: 1
- fasttext_minn: 2
- fasttext_maxn: 5
- fasttext_dim: 256
- fasttext_bucket: 1000000
- fasttext_minCount: 500
- fasttext_loss: softmax
- fasttext_thread: 48

## Experiment: veracruz_large_wordNgrams2
- date: 2026-03-04 11:36 UTC
- script: train_veracruz.py
- In-domain PT-PT F1: 88.2%
- DSL-TL PT-PT F1: 71.7%
- FRMT PT-PT F1: 75.6%
- name: veracruz_large_wordNgrams2
- fasttext_lr: 0.8
- fasttext_epoch: 3
- fasttext_wordNgrams: 2
- fasttext_minn: 2
- fasttext_maxn: 5
- fasttext_dim: 256
- fasttext_bucket: 1000000
- fasttext_minCount: 500
- fasttext_loss: softmax
- fasttext_thread: 48

## Experiment: veracruz_large_wordNgrams2epochs5
- date: 2026-03-04 12:14 UTC
- script: train_veracruz.py
- In-domain PT-PT F1: 88.4%
- DSL-TL PT-PT F1: 71.8%
- FRMT PT-PT F1: 75.7%
- name: veracruz_large_wordNgrams2epochs5
- fasttext_lr: 0.8
- fasttext_epoch: 5
- fasttext_wordNgrams: 2
- fasttext_minn: 2
- fasttext_maxn: 5
- fasttext_dim: 256
- fasttext_bucket: 1000000
- fasttext_minCount: 500
- fasttext_loss: softmax
- fasttext_thread: 48

## Experiment: veracruz_autotune
- date: 2026-03-04 13:36 UTC
- script: train_veracruz_optimized.py
- In-domain PT-PT F1: 87.5%
- DSL-TL PT-PT F1: 69.0%
- FRMT PT-PT F1: 73.3%
- name: veracruz_autotune
- autotune_duration: 600
- autotune_metric: f1:__label__PT_PT
- selected_params: {'lr': 0.1, 'epoch': 5, 'wordNgrams': 1, 'minn': 0, 'maxn': 0, 'dim': 100, 'bucket': 0, 'minCount': 1, 'loss': 'softmax'}

## Experiment: veracruz_6M_wordNgrams2epochs5
- date: 2026-03-04 15:21 UTC
- script: train_veracruz.py
- In-domain PT-PT F1: 89.0%
- DSL-TL PT-PT F1: 71.6%
- FRMT PT-PT F1: 76.6%
- name: veracruz_6M_wordNgrams2epochs5
- fasttext_lr: 0.8
- fasttext_epoch: 5
- fasttext_wordNgrams: 2
- fasttext_minn: 2
- fasttext_maxn: 5
- fasttext_dim: 256
- fasttext_bucket: 1000000
- fasttext_minCount: 500
- fasttext_loss: hs
- fasttext_thread: 48

## Experiment: veracruz_6M_wordNgrams2epochs5
- date: 2026-03-04 15:43 UTC
- script: train_veracruz.py
- In-domain PT-PT F1: 82.6%
- DSL-TL PT-PT F1: 65.5%
- FRMT PT-PT F1: 70.3%
- name: veracruz_6M_wordNgrams2epochs5
- fasttext_lr: 0.8
- fasttext_epoch: 5
- fasttext_wordNgrams: 2
- fasttext_minn: 2
- fasttext_maxn: 5
- fasttext_dim: 256
- fasttext_bucket: 1000000
- fasttext_minCount: 500
- fasttext_loss: hs
- fasttext_thread: 48

## Experiment: tiny_baseline
- date: 2026-03-04 15:48 UTC
- script: train_veracruz.py
- In-domain PT-PT F1: 82.7%
- DSL-TL PT-PT F1: 65.8%
- FRMT PT-PT F1: 69.4%
- name: tiny_baseline
- preprocess: False
- fasttext_lr: 0.8
- fasttext_epoch: 5
- fasttext_wordNgrams: 2
- fasttext_minn: 2
- fasttext_maxn: 5
- fasttext_dim: 256
- fasttext_bucket: 1000000
- fasttext_minCount: 500
- fasttext_loss: hs
- fasttext_thread: 48

## Experiment: tiny_preprocess
- date: 2026-03-04 15:49 UTC
- script: train_veracruz.py
- In-domain PT-PT F1: 82.5%
- DSL-TL PT-PT F1: 66.5%
- FRMT PT-PT F1: 70.3%
- name: tiny_preprocess
- preprocess: True
- fasttext_lr: 0.8
- fasttext_epoch: 5
- fasttext_wordNgrams: 2
- fasttext_minn: 2
- fasttext_maxn: 5
- fasttext_dim: 256
- fasttext_bucket: 1000000
- fasttext_minCount: 500
- fasttext_loss: hs
- fasttext_thread: 48

## Experiment: tiny_lr_0.1
- date: 2026-03-04 15:49 UTC
- script: train_veracruz.py
- In-domain PT-PT F1: 77.9%
- DSL-TL PT-PT F1: 63.8%
- FRMT PT-PT F1: 65.5%
- name: tiny_lr_0.1
- preprocess: False
- fasttext_lr: 0.1
- fasttext_epoch: 5
- fasttext_wordNgrams: 2
- fasttext_minn: 2
- fasttext_maxn: 5
- fasttext_dim: 256
- fasttext_bucket: 1000000
- fasttext_minCount: 500
- fasttext_loss: hs
- fasttext_thread: 48

## Experiment: tiny_lr_0.5
- date: 2026-03-04 15:50 UTC
- script: train_veracruz.py
- In-domain PT-PT F1: 82.7%
- DSL-TL PT-PT F1: 65.5%
- FRMT PT-PT F1: 70.1%
- name: tiny_lr_0.5
- preprocess: False
- fasttext_lr: 0.5
- fasttext_epoch: 5
- fasttext_wordNgrams: 2
- fasttext_minn: 2
- fasttext_maxn: 5
- fasttext_dim: 256
- fasttext_bucket: 1000000
- fasttext_minCount: 500
- fasttext_loss: hs
- fasttext_thread: 48

## Experiment: tiny_lr_1.5
- date: 2026-03-04 15:50 UTC
- script: train_veracruz.py
- In-domain PT-PT F1: 82.9%
- DSL-TL PT-PT F1: 66.2%
- FRMT PT-PT F1: 69.2%
- name: tiny_lr_1.5
- preprocess: False
- fasttext_lr: 1.5
- fasttext_epoch: 5
- fasttext_wordNgrams: 2
- fasttext_minn: 2
- fasttext_maxn: 5
- fasttext_dim: 256
- fasttext_bucket: 1000000
- fasttext_minCount: 500
- fasttext_loss: hs
- fasttext_thread: 48

## Experiment: tiny_lr_3.0
- date: 2026-03-04 15:51 UTC
- script: train_veracruz.py
- In-domain PT-PT F1: 82.6%
- DSL-TL PT-PT F1: 65.5%
- FRMT PT-PT F1: 70.1%
- name: tiny_lr_3.0
- preprocess: False
- fasttext_lr: 3.0
- fasttext_epoch: 5
- fasttext_wordNgrams: 2
- fasttext_minn: 2
- fasttext_maxn: 5
- fasttext_dim: 256
- fasttext_bucket: 1000000
- fasttext_minCount: 500
- fasttext_loss: hs
- fasttext_thread: 48

## Experiment: tiny_epoch_2
- date: 2026-03-04 15:51 UTC
- script: train_veracruz.py
- In-domain PT-PT F1: 81.3%
- DSL-TL PT-PT F1: 65.1%
- FRMT PT-PT F1: 68.6%
- name: tiny_epoch_2
- preprocess: False
- fasttext_lr: 0.8
- fasttext_epoch: 2
- fasttext_wordNgrams: 2
- fasttext_minn: 2
- fasttext_maxn: 5
- fasttext_dim: 256
- fasttext_bucket: 1000000
- fasttext_minCount: 500
- fasttext_loss: hs
- fasttext_thread: 48

## Experiment: tiny_epoch_10
- date: 2026-03-04 15:52 UTC
- script: train_veracruz.py
- In-domain PT-PT F1: 82.5%
- DSL-TL PT-PT F1: 64.8%
- FRMT PT-PT F1: 69.7%
- name: tiny_epoch_10
- preprocess: False
- fasttext_lr: 0.8
- fasttext_epoch: 10
- fasttext_wordNgrams: 2
- fasttext_minn: 2
- fasttext_maxn: 5
- fasttext_dim: 256
- fasttext_bucket: 1000000
- fasttext_minCount: 500
- fasttext_loss: hs
- fasttext_thread: 48

## Experiment: veracruz_6M_preprocess
- date: 2026-03-04 17:00 UTC
- script: train_veracruz.py
- In-domain PT-PT F1: 88.6%
- DSL-TL PT-PT F1: 71.8%
- FRMT PT-PT F1: 77.4%
- name: veracruz_6M_preprocess
- preprocess: True
- fasttext_lr: 0.8
- fasttext_epoch: 5
- fasttext_wordNgrams: 2
- fasttext_minn: 2
- fasttext_maxn: 5
- fasttext_dim: 256
- fasttext_bucket: 1000000
- fasttext_minCount: 500
- fasttext_loss: hs
- fasttext_thread: 48

Confusion matrix:

DSL-TL:
              precision    recall  f1-score   support

       PT-PT       0.59      0.91      0.72       269
       PT-BR       0.95      0.71      0.81       588

    accuracy                           0.78       857
   macro avg       0.77      0.81      0.77       857
weighted avg       0.84      0.78      0.78       857

  PT-PT F1: 71.8% (vs. 75% best)

Loading FRMT test...

FRMT:
              precision    recall  f1-score   support

       PT-PT       0.69      0.88      0.77      2614
       PT-BR       0.83      0.61      0.70      2612

    accuracy                           0.74      5226
   macro avg       0.76      0.74      0.74      5226
weighted avg       0.76      0.74      0.74      5226

  PT-PT F1: 77.4% (vs. 76% best)

