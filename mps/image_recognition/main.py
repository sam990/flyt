import argparse
import subprocess
import command
import os

parser = argparse.ArgumentParser()

parser.add_argument('--limit-core-percentage', default=100, type=str, action="store", dest="limit_cores", help="Mention the utilization limit for the program, for e.g. --limit-core-percentage 20")
parser.add_argument('--limit-memory', default="'0=6G'", type=str, action="store", dest="limit_memory", help="Mention the amount of memory that would be limited for all programs, for e.g. --limit-memory '0=5G' ")
#parser.add_argument('--image-recognition', default="'0=6G'", type=str, action="store", dest="limit_memory", help="Mention the image recognition program to run. --image-recognition resnet.py")
arguments = parser.parse_args()
limit_cores = arguments.limit_cores
limit_memory = arguments.limit_memory
#print(limit_memory)
os.environ["CUDA_MPS_PINNED_DEVICE_MEM_LIMIT"] = limit_memory
os.environ["CUDA_MPS_ACTIVE_THREAD_PERCENTAGE"] = limit_cores
#os.system('echo $CUDA_MPS_PINNED_DEVICE_MEM_LIMIT')

#os.system()

#p = subprocess.call(f"export CUDA_MPS_PINNED_DEVICE_MEM_LIMIT='{limit_memory}'" , shell=True)
#res = command.run(["export CUDA_MPS_PINNED_DEVICE_MEM_LIMIT='"+limit_memory+"'"])
