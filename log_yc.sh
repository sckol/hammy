function log_yc() {
  source /root/.bashrc
  script="$1"
  shift

  {
    # Capture stderr first and redirect stdout to a pipe (fd 3)
    "$script" "$@" | while IFS= read -r line; do
      echo "Info log: $line" >&3
      yc logging write \
        --group-name=hammy-compute \
        --message="$line" \
        --level=INFO        
    done
  } 3>&2 2>&1 | while IFS= read -r error_line; do
    echo "Error log: $error_line"
    yc logging write \
        --group-name=hammy-compute \
        --message="$error_line" \
        --level=ERROR        
  done
} 3>&1 1>&2 2>&3
