#!/bin/bash

# Exit on error
set -e

echo "========================================"
echo "      Starting Phase 0 Execution        "
echo "========================================"

# Activate venv
source venv/bin/activate

# 1. Generate Data
echo ""
echo "[1/4] Generating FENs..."
python scripts/generate_fens.py

# 2. Audit Tokenizer
echo ""
echo "[2/4] Auditing Tokenizer..."
python -u scripts/audit_tokenizer.py > results/audit_log.txt 2>&1
echo "Audit complete. Results in results/audit_log.txt"
cat results/audit_log.txt

# 3. Prompt Search
echo ""
echo "[3/4] Running Prompt Search..."
python -u scripts/prompt_search.py > results/prompt_search_log.txt 2>&1
echo "Prompt Search complete. Results in results/prompt_search_log.txt"
tail -n 10 results/prompt_search_log.txt

# 4. Benchmark Models
echo ""
echo "[4/4] Running Model Benchmark..."
python -u scripts/benchmark_models.py > results/benchmark_log.txt 2>&1
echo "Benchmark complete. Results in results/benchmark_log.txt"
tail -n 10 results/benchmark_log.txt

echo ""
echo "========================================"
echo "      Phase 0 Execution Complete        "
echo "========================================"
