 # Loop 7 times
 for run in {1..2}; do
  echo "Killing"
  ps aux | grep python | grep eval.py | awk '{print $2}' |  xargs kill
  echo "Sleep"
  sleep 5
done

