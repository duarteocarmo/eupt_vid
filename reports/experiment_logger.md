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
