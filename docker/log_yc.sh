#!/bin/bash
log_line() {
  line="$1"
  shared_file="$2"
  level="$3"  
  source "$shared_file"
  current_timestamp=$(date +%s)
  make_log=false  
  if (( current_timestamp - last_call[$level] >= timeout )); then
    c_local[$level]=1
    make_log=true
  elif (( c_local[$level] < max_lines )); then      
    ((c_local[$level]++))
    make_log=true
  fi  
  if [ "$make_log" = true ]; then
    ((c_global++))
    last_call[$level]="$current_timestamp"
    printf "c_global=$c_global\n$(declare -p c_local)\n$(declare -p last_call)" > "$shared_file"
    if ((c_local[$level] == max_lines )); then
      postfix="...~"
    else
      postfix=""
    fi       
    line+="$postfix"                               
    msg_echo=$(printf "(%06d) %s log: %s" "$c_global" "$level" "$line")
    msg_yc=$(printf "(%06d) %s" "$c_global" "$line")
    echo "$msg_echo"
    if [ ! "$DISABLE_YC_LOG" = true ]; then
      yc logging write \
        --group-name=hammy-compute \
        --message="$msg_yc" \
        --level=$level \
        --resource-id="$YC_INSTANCE_ID"
    fi
  fi        
}

log_yc() {
  shared_file=$(mktemp)
  timeout="${LYC_T:-5}"
  max_lines="${LYC_ML:-5}"
  script="$1"
  shift
  printf "c_global=0\ndeclare -A c_local=( [INFO]=0 [ERROR]=0 )\ndeclare -A last_call=( [ERROR]=0 [INFO]=0 )" > "$shared_file"
  {   
    "$script" "$@" | while IFS= read -r line; do
      log_line "$line" "$shared_file" INFO >&3
    done
  } 3>&2 2>&1 | while IFS= read -r line; do
      log_line "$line" "$shared_file" ERROR
    done  
  rm "$shared_file"
} 3>&1 1>&2 2>&3