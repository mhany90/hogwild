#!/bin/bash
echo "One worker"
python3 mlponeworker.py --job_name="ps" --task_index=0 & python3 mlponeworker.py --job_name="worker" --task_index=0

wait %%

Echo "Two workers"
python3 mlp6.py --job_name="ps" --task_index=0 & python3 mlp6.py --job_name="worker" --task_index=0 & python3 mlp6.py --job_name="worker" --task_index=1
