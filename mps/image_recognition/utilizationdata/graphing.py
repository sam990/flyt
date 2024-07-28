time_intervals=[]
gpu_utils=[]
with open("gpu_util_with_sus_200ms.txt","r") as fd:
    for ind,data in enumerate(fd,1):
        time_intervals.append(ind*100)
        temp=data[:-2]
        gpu_utils.append(temp)
print(gpu_utils)
import matplotlib.pyplot as plt
plt.plot(time_intervals,gpu_utils,marker='o')
#plt.show()
