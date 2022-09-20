import GPUtil
import time
import os
FINISHED = 0 # The finished tasks number.
EXCLUDE = [] # The GPU ID to exclude.
sparse_list = [95,85,75,65] # The sparse rate, should included in datasets.

# Write the command line to run the tested models. {0} is the GPU ID which is decided automatically, {1} is the sparse rate.
task_list = [
# betaE
'CUDA_VISIBLE_DEVICES={0} nohup python KGReasoning/main.py --cuda --do_train --do_test --sparse_test --data_path data/family_gene/q2b/{1} --cpu_num 4 --max_steps 400000 --valid_steps 300 --geo beta --tasks 1p --seed 0 --no_valid_train -d 500 -g 60 --label 0 &',
'CUDA_VISIBLE_DEVICES={0} nohup python KGReasoning/main.py --cuda --do_train --do_test --sparse_test --data_path data/family_gene/q2b/{1} --cpu_num 4 --max_steps 400000 --valid_steps 300 --geo beta --tasks 1p --seed 1 --no_valid_train -d 100 -g 60 --label 1 &',
'CUDA_VISIBLE_DEVICES={0} nohup python KGReasoning/main.py --cuda --do_train --do_test --sparse_test --data_path data/family_gene/q2b/{1} --cpu_num 4 --max_steps 400000 --valid_steps 300 --geo beta --tasks 1p --seed 2 --no_valid_train -d 1000 -g 60 --label 2 &',
'CUDA_VISIBLE_DEVICES={0} nohup python KGReasoning/main.py --cuda --do_train --do_test --sparse_test --data_path data/family_gene/q2b/{1} --cpu_num 4 --max_steps 400000 --valid_steps 300 --geo beta --tasks 1p --seed 3 --no_valid_train -d 500 -g 15 --label 3 &',
'CUDA_VISIBLE_DEVICES={0} nohup python KGReasoning/main.py --cuda --do_train --do_test --sparse_test --data_path data/family_gene/q2b/{1} --cpu_num 4 --max_steps 400000 --valid_steps 300 --geo beta --tasks 1p --seed 4 --no_valid_train -d 500 -g 240 --label 4 &',
# RotatE
'CUDA_VISIBLE_DEVICES={0} nohup python kge-master/codes/run.py --cuda --do_train --do_test --data_path data/family_gene/transe/{1} --save_path logs/{1}/rotate/0 --cpu_num 4 --max_steps 100000 --valid_steps 100 --model RotatE -de --seed 10 -g 24 &',
'CUDA_VISIBLE_DEVICES={0} nohup python kge-master/codes/run.py --cuda --do_train --do_test --data_path data/family_gene/transe/{1} --save_path logs/{1}/rotate/1 --cpu_num 4 --max_steps 100000 --valid_steps 100 --model RotatE -de --seed 11 -g 24 -d 1000 &',
'CUDA_VISIBLE_DEVICES={0} nohup python kge-master/codes/run.py --cuda --do_train --do_test --data_path data/family_gene/transe/{1} --save_path logs/{1}/rotate/2 --cpu_num 4 --max_steps 100000 --valid_steps 100 --model RotatE -de --seed 12 -g 24 -d 100 &',
'CUDA_VISIBLE_DEVICES={0} nohup python kge-master/codes/run.py --cuda --do_train --do_test --data_path data/family_gene/transe/{1} --save_path logs/{1}/rotate/3 --cpu_num 4 --max_steps 100000 --valid_steps 100 --model RotatE -de --seed 16 -b 256 &',
'CUDA_VISIBLE_DEVICES={0} nohup python kge-master/codes/run.py --cuda --do_train --do_test --data_path data/family_gene/transe/{1} --save_path logs/{1}/rotate/4 --cpu_num 4 --max_steps 100000 --valid_steps 100 --model RotatE -de --seed 17 -n 512 &',
# ComplEx
'CUDA_VISIBLE_DEVICES={0} nohup python kge-master/codes/run.py --cuda --do_train --do_test --data_path data/family_gene/transe/{1} --save_path logs/{1}/complex/0 --cpu_num 4 --max_steps 100000 --valid_steps 300 --model ComplEx -de -dr --seed 20 -g 500 &',
'CUDA_VISIBLE_DEVICES={0} nohup python kge-master/codes/run.py --cuda --do_train --do_test --data_path data/family_gene/transe/{1} --save_path logs/{1}/complex/1 --cpu_num 4 --max_steps 100000 --valid_steps 300 --model ComplEx -de -dr --seed 21 -g 500 -d 1000 &',
'CUDA_VISIBLE_DEVICES={0} nohup python kge-master/codes/run.py --cuda --do_train --do_test --data_path data/family_gene/transe/{1} --save_path logs/{1}/complex/2 --cpu_num 4 --max_steps 100000 --valid_steps 300 --model ComplEx -de -dr --seed 23 -g 200 -n 256 -b 512 &',
# pRotatE
'CUDA_VISIBLE_DEVICES={0} nohup python kge-master/codes/run.py --cuda --do_train --do_test --data_path data/family_gene/transe/{1} --save_path logs/{1}/protate/0 --cpu_num 4 --max_steps 12000 --valid_steps 50 --model pRotatE --seed 30 -g 24 &',
'CUDA_VISIBLE_DEVICES={0} nohup python kge-master/codes/run.py --cuda --do_train --do_test --data_path data/family_gene/transe/{1} --save_path logs/{1}/protate/1 --cpu_num 4 --max_steps 12000 --valid_steps 50 --model pRotatE --seed 31 -g 6 &',
'CUDA_VISIBLE_DEVICES={0} nohup python kge-master/codes/run.py --cuda --do_train --do_test --data_path data/family_gene/transe/{1} --save_path logs/{1}/protate/2 --cpu_num 4 --max_steps 12000 --valid_steps 50 --model pRotatE --seed 32 -g 24 -d 1000 &',
'CUDA_VISIBLE_DEVICES={0} nohup python kge-master/codes/run.py --cuda --do_train --do_test --data_path data/family_gene/transe/{1} --save_path logs/{1}/protate/3 --cpu_num 4 --max_steps 12000 --valid_steps 50 --model pRotatE --seed 33 -g 24 -d 500 -b 128 -n 512 &', # new one
'CUDA_VISIBLE_DEVICES={0} nohup python kge-master/codes/run.py --cuda --do_train --do_test --data_path data/family_gene/transe/{1} --save_path logs/{1}/protate/4 --cpu_num 4 --max_steps 12000 --valid_steps 50 --model pRotatE --seed 34 -g 24 -d 250 &', # new one
]
task_num = len(sparse_list) * len(task_list)
with open('auto.log', 'w') as log:
    log.write('Total num {}. Have finished {}.\n'.format(task_num, FINISHED))
last_deviceIDs = []
lastlast_deviceIDs = []
for sparse in sparse_list:
    for task in task_list:
        if FINISHED:
            FINISHED -= 1
            task_num -= 1
            continue
        while True:
            deviceIDs = GPUtil.getAvailable(maxLoad = 0.95, maxMemory = 0.6, includeNan=False, excludeID=EXCLUDE, excludeUUID=[])
            time.sleep(3)
            available = list(set(lastlast_deviceIDs).intersection(set(last_deviceIDs).intersection(set(deviceIDs))))
            lastlast_deviceIDs = last_deviceIDs
            last_deviceIDs = deviceIDs
            if available:
                last_deviceIDs = []
                lastlast_deviceIDs = []
                break
        GPUID = available[0]
        task = task.format(GPUID, sparse)
        cur_time = time.strftime("%Y.%m.%d-%H:%M:%S", time.localtime())
        os.system(task)
        task_num -= 1
        with open('auto.log', 'a') as log:
            log.write('{}\n{}\nResidual num: {}\n\n'.format(cur_time, task, task_num))
