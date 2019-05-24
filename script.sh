sudo kill -9 $(ps aux | grep 877 | awk  '{print $2}')
