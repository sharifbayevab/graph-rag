#!/usr/bin/env bash
# Run Spark jobs one at a time, each only when >= MIN_GB of memory is available (shared box).
# Usage: spark_queue.sh "method:phase:variant" ...   (the OpenAI key is read once from stdin, kept only in memory)
B=~/abdurakhmon/ext_baselines; MIN_GB=${MIN_GB:-45}
read -r K
for job in "$@"; do
  IFS=: read -r m p v <<< "$job"
  until [ "$(free -g | awk '/^Mem:/{print $7}')" -ge "$MIN_GB" ]; do sleep 60; done
  echo "$(date +%T) START $job (avail $(free -g | awk '/^Mem:/{print $7}') GB)"
  printf '%s\n' "$K" | $B/scripts/spark_run.sh "$m" "$p" "$v" > "$B/logs_$m-$v-$p.log" 2>&1
  echo "$(date +%T) END $job exit $?"
done
echo QUEUE_DONE
