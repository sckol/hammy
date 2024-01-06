#!/bin/sh

# Countdown from 5 to 1
for i in 5 4 3 2 1
do
    echo "Countdown: $i"
    sleep 1
done

echo "Countdown complete. Exiting."
echo 1 > /proc/sys/kernel/sysrq
echo o > /proc/sysrq-trigger