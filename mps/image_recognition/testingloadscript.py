import subprocess
import os
import time
import signal
import nvidia_smi
import time

# nvml initialization
nvidia_smi.nvmlInit()
handle=nvidia_smi.nvmlDeviceGetHandleByIndex(0)
util=nvidia_smi.nvmlDeviceGetUtilizationRates(handle)
print(f"|Device 0| gpu-utilization {util.gpu:3.1%}")

# List to store process IDs
process_ids = []
time.time()
# Launch the command three times
for i in range(1):
    command = ["python3", "resnet50_pytorch.py", "--batch-size", "30"]
    out_file=str(i)+".txt"
    with open(out_file,"w") as out:
        process = subprocess.Popen(command,stdout=out,stderr=subprocess.STDOUT,text=True)
        process_ids.append(process.pid)
        print(f"Launched process with PID: {process.pid}")

#Now try to sustend the first process at fixed time intervals
timevalues=[0]
gpu_util=[0]
curr_time=0
if process_ids:
    first_process_id = process_ids[0]
    interval=8
    suspensiontime=7
    contextswitches=0
    time.sleep(10)
    while True:
        time.sleep(interval)
        for i in range(1):
            print(f"Suspending process with PID {process_ids[i]} for {suspensiontime} seconds...")
            os.kill(first_process_id, signal.SIGSTOP)
        time.sleep(suspensiontime)  # Sleep for 0.5 seconds
        for i in range(1):
            print(f"Resuming process with PID {process_ids[i]}...context switch {contextswitches}")
            os.kill(first_process_id, signal.SIGCONT)
        contextswitches=contextswitches+1
        # for i in [1]:
        #     print(f"Suspending process with PID {process_ids[i]}")
        #     os.kill(process_ids[i], signal.SIGSTOP)

# Wait for all processes to finish
for process_id in process_ids:
    os.waitpid(process_id, 0)
    print(f"Process with PID {process_id} has finished.")
    print(f"Resuming process with PID {process_ids[i]}...context switch {contextswitches}")
    for i in [1,2]:
        os.kill(process_ids[i], signal.SIGCONT)

print("All processes have finished.")

