#!/bin/bash
run_nice () {
    (echo 1000 > /proc/self/oom_score_adj && exec nice -n 10 "$@")
}