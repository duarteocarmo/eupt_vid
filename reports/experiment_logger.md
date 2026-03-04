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


# Experiment Log

| Date | Experiment | Script | Metrics |
|------|------------|--------|---------|
| 2026-03-04 10:13 UTC | journalistic_baseline | `train_journalistic.py` | In-domain PT-PT F1: 97.4%, DSL-TL PT-PT F1: 65.3%, FRMT PT-PT F1: 47.0% |
| 2026-03-04 10:21 UTC | veracruz_baseline | `train_veracruz.py` | In-domain PT-PT F1: 86.6%, DSL-TL PT-PT F1: 70.5%, FRMT PT-PT F1: 72.9% |

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
