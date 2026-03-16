# Multi-Seed Sweep

Seeds: 7, 11, 19
Best PPO seed: 19

## Aggregate Metrics

| Policy | Success mean±std | Distance mean±std | Effort mean±std | Assist mean±std | Fatigue mean±std |
| --- | --- | --- | --- | --- | --- |
| ppo_rl | 0.592±0.125 | 0.508±0.012 | 0.285±0.002 | 0.141±0.020 | 0.635±0.017 |
| zero_assist | 0.283±0.042 | 0.603±0.014 | 0.285±0.009 | 0.000±0.000 | 0.669±0.010 |
| fixed_assist | 0.450±0.102 | 0.564±0.025 | 0.288±0.004 | 0.113±0.004 | 0.651±0.013 |
| rule_based_fatigue | 0.783±0.105 | 0.507±0.027 | 0.290±0.003 | 0.267±0.007 | 0.643±0.006 |

## Per-Seed PPO

| Seed | Success | Distance | Effort | Assist | Fatigue | Score |
| --- | --- | --- | --- | --- | --- | --- |
| 7 | 0.625 | 0.524 | 0.284 | 0.168 | 0.651 | 1.289 |
| 11 | 0.425 | 0.505 | 0.283 | 0.120 | 0.643 | 0.804 |
| 19 | 0.725 | 0.494 | 0.288 | 0.136 | 0.612 | 1.558 |
