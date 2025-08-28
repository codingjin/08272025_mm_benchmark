#!/bin/bash

if [ $# -eq 0 ]; then
    echo "No arguments provided. Usage: $0 <CPU>"
    echo "Valid CPU values: r9, i7, rt"
    exit 1
elif [ $# -gt 1 ]; then
    echo "Too many arguments. Only one argument <CPU> is expected."
    exit 1
fi

cpu="$1"
# Check validity
if [[ "$cpu" != "r9" && "$cpu" != "i7" && "$cpu" != "rt" ]]; then
    echo "Invalid input. Please enter one of: r9, i7, rt."
    exit 1
fi

make
mkdir -p $cpu

n=4096
i=1
while [ $i -le $n ]; do
    ./openblas_sgemm 4096 4096 "$i" "$cpu" 2>&1 | tee "$cpu/4096_4096_$i"
    i=$(( i * 2 ))
done

