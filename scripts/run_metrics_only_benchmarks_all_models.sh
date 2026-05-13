RUN_NAME=images_all BENCHMARK_MODE=metrics_only SAMPLE_LIMIT=1000 METRICS_BATCH_SIZE=50 KID_BATCH_SIZE=50 KID_SUBSET_SIZE=1000 bash scripts/run_ddcolor_benchmark.sh --json_name report_1000.json --csv_name summary_1000.csv
RUN_NAME=images_all BENCHMARK_MODE=metrics_only SAMPLE_LIMIT=1000 METRICS_BATCH_SIZE=50 KID_BATCH_SIZE=50 KID_SUBSET_SIZE=1000 bash scripts/run_deoldify_benchmark.sh --json_name report_1000.json --csv_name summary_1000.csv
RUN_NAME=images_all BENCHMARK_MODE=metrics_only SAMPLE_LIMIT=1000 METRICS_BATCH_SIZE=50 KID_BATCH_SIZE=50 KID_SUBSET_SIZE=1000 bash scripts/run_colorcomic_auto_benchmark.sh --json_name report_1000.json --csv_name summary_1000.csv
RUN_NAME=images_all BENCHMARK_MODE=metrics_only SAMPLE_LIMIT=1000 METRICS_BATCH_SIZE=50 KID_BATCH_SIZE=50 KID_SUBSET_SIZE=1000 REFERENCE_MODE=fixed_by_title bash scripts/run_cgan_reference_benchmark.sh --json_name report_1000.json --csv_name summary_1000.csv
RUN_NAME=images_all BENCHMARK_MODE=metrics_only SAMPLE_LIMIT=1000 METRICS_BATCH_SIZE=50 KID_BATCH_SIZE=50 KID_SUBSET_SIZE=1000 REFERENCE_MODE=previous_output_by_title bash scripts/run_cgan_reference_benchmark.sh --json_name report_1000.json --csv_name summary_1000.csv

RUN_NAME=images_all BENCHMARK_MODE=metrics_only METRICS_BATCH_SIZE=50 KID_BATCH_SIZE=50 KID_SUBSET_SIZE=1000 bash scripts/run_ddcolor_benchmark.sh --json_name report_all.json --csv_name summary_all.csv
RUN_NAME=images_all BENCHMARK_MODE=metrics_only METRICS_BATCH_SIZE=50 KID_BATCH_SIZE=50 KID_SUBSET_SIZE=1000 bash scripts/run_deoldify_benchmark.sh --json_name report_all.json --csv_name summary_all.csv
RUN_NAME=images_all BENCHMARK_MODE=metrics_only METRICS_BATCH_SIZE=50 KID_BATCH_SIZE=50 KID_SUBSET_SIZE=1000 bash scripts/run_colorcomic_auto_benchmark.sh --json_name report_all.json --csv_name summary_all.csv
RUN_NAME=images_all BENCHMARK_MODE=metrics_only METRICS_BATCH_SIZE=50 KID_BATCH_SIZE=50 KID_SUBSET_SIZE=1000 REFERENCE_MODE=fixed_by_title bash scripts/run_cgan_reference_benchmark.sh --json_name report_all.json --csv_name summary_all.csv
RUN_NAME=images_all BENCHMARK_MODE=metrics_only METRICS_BATCH_SIZE=50 KID_BATCH_SIZE=50 KID_SUBSET_SIZE=1000 REFERENCE_MODE=previous_output_by_title bash scripts/run_cgan_reference_benchmark.sh --json_name report_all.json --csv_name summary_all.csv

