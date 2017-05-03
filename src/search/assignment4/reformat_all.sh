rm -r *.p

chmod +x assignment4/reformatter.py

python3.6 -m assignment4.reformatter assignment2/info_ret.xml --job_path=assignment4/df_jobs --num_partitions=5

