# export PYDEVD_WARN_SLOW_RESOLVE_TIMEOUT=1
rm -r output
rm -r belief
rm video/*
rm test.log

# nohup python src/main.py > test.log 2>&1 &