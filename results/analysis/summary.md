# Multi-LoRA Benchmark Summary

## Experiment: hot_change

| system | hot_ratio | num_requests | mean_latency_ms | p99_latency_ms | throughput (req/s) | notes |
| --- | --- | --- | --- | --- | --- | --- |
| dynamic_multi_lora | - | 1199 | 219.0 | 1298.8 | 2.00 | num_switch_commits=1, fast_path_share=0.6905754795663053 |
| multi_lora | - | 1200 | 316.9 | 1108.6 | 2.00 | initial_hot=hot_general, second_hot=cold_dummy, num_failed=0, source_file=lru_E2_hot_change_hot_general_to_cold_dummy.json |

## Experiment: stable_skew

| system | hot_ratio | num_requests | mean_latency_ms | p99_latency_ms | throughput (req/s) | notes |
| --- | --- | --- | --- | --- | --- | --- |
| dynamic_multi_lora | 0.00 | 100 | 299.2 | 1150.3 | 3.34 | fast_path_share=0.0, num_switch_commits=0 |
| dynamic_multi_lora | 0.40 | 100 | 188.5 | 1129.0 | 5.30 | fast_path_share=0.4, num_switch_commits=0 |
| dynamic_multi_lora | 0.80 | 100 | 147.1 | 1127.0 | 6.79 | fast_path_share=0.8, num_switch_commits=0 |
| dynamic_multi_lora | 1.00 | 100 | 103.2 | 317.7 | 9.67 | fast_path_share=1.0, num_switch_commits=0 |
| multi_lora | 0.00 | 200 | 604.5 | 1103.3 | 1.65 | errors=0, empty_text_responses=41 |
| multi_lora | 0.20 | 200 | 544.6 | 1101.4 | 1.84 | errors=0, empty_text_responses=49 |
| multi_lora | 0.40 | 200 | 453.2 | 1100.1 | 2.21 | errors=0, empty_text_responses=61 |
| multi_lora | 0.60 | 200 | 396.1 | 1103.9 | 2.52 | errors=0, empty_text_responses=54 |
| multi_lora | 0.80 | 200 | 335.3 | 1101.7 | 2.98 | errors=0, empty_text_responses=52 |
| multi_lora | 1.00 | 200 | 288.4 | 1093.5 | 3.47 | errors=0, empty_text_responses=53 |
| static_premerged[hot=hot_general] | 0.00 | 200 | 253.3 | 1183.7 | 3.95 | errors=0 |
| static_premerged[hot=hot_general] | 0.20 | 200 | 238.2 | 1178.6 | 4.20 | errors=0 |
| static_premerged[hot=hot_general] | 0.40 | 200 | 214.2 | 1182.7 | 4.66 | errors=0 |
| static_premerged[hot=hot_general] | 0.60 | 200 | 200.5 | 1179.6 | 4.99 | errors=0 |
| static_premerged[hot=hot_general] | 0.80 | 200 | 143.9 | 1174.8 | 6.94 | errors=0 |
| static_premerged[hot=hot_general] | 1.00 | 200 | 116.0 | 380.0 | 8.61 | errors=0 |

## Experiment: thrashing

| system | hot_ratio | num_requests | mean_latency_ms | p99_latency_ms | throughput (req/s) | notes |
| --- | --- | --- | --- | --- | --- | --- |
| dynamic_multi_lora[thrash_guarded] | - | 360 | 223.1 | 1070.7 | 2.00 | num_switch_commits=0 |
| dynamic_multi_lora[thrash_naive] | - | 360 | 257.7 | 1242.1 | 2.00 | num_switch_commits=3 |
