check_results=`pip show bert4kerasd | grep "Version"`

echo "command(rpm -qa) results are: $check_results"

if [ ! -n "$check_results" ]; then
  pip install git+https://www.github.com/bojone/bert4keras.git@v0.2.6
else
  echo "NOT NULL"
fi