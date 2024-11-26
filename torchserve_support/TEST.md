# Goal
To run a production-grade inference server in a VM that:
i. Exposes a REST endpoint to client machines, from where it receives inference requests and data.
ii. Returns the results back to clients after evaulating on GPU given a pre-trained model archive (.mar)

# Installation
Pre-requisite: Flyt VM with Pytorch support.
```bash 
pip install torchserve torch-model-archiver torch-workflow-archiver captum
```

# Quickstart
a. Create a torchserve model in a custom directory in a Flyt VM.
```bash
mkdir model_store

# A serialized .pth file is required for each model.
wget https://download.pytorch.org/models/vgg16-397923af.pth

# A .mar file needs to be created for each model that
# your instance must serve.
torch-model-archiver --model-name vgg16 --version 1.0 --model-file ./model.py --serialized-file ./vgg16-397923af.pth --handler ./vgg_handler.py  --config-file ./model-config.yaml -f

mv vgg16.mar model_store/
```

b. Create a torchserve configuration file `config.properties` to specify 
IP, port, number of backend workers spawned, etc.
```yaml
# File: config.properties
# -----------------------
# Specify torchserve configuration.
 
inference_address=http://10.129.28.180:8080
management_address=http://10.129.28.180:8081
metrics_address=http://10.129.28.180:8082

min_workers=1
max_workers=1
default_workers_per_model=1
```

c. Start the inference server in your Flyt VM.
```bash
# From the directory containing model_store
torchserve --start --ncs --model-store model_store --models vgg16=vgg16.mar --ts-config ./vgg16/config.properties --disable-token-auth  --enable-model-api

# To stop the server
torchserve --stop
```

d. Send inference requests to the torchserve instance.
Requests are sent via curl from any machine in the LAN of your Flyt VM.
```bash
# FROM A DIFFERENT MACHINE
# ------------------------

# Test frontend Health
curl http://10.129.28.123:8080/ping 

# Get image test data
curl -O https://raw.githubusercontent.com/pytorch/serve/master/docs/images/kitten_small.jpg

# Send one inference request to server
curl http://10.129.28.180:8080/predictions/vgg16 -T kitten_small.jpg
```

# Benchmark


