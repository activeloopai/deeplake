#!/bin/bash
for filename in ./benchmark_*.py; do
    [ -e "$filename" ] || continue
    python3 ./$filename >> results.log
done