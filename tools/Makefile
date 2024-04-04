all: get-gpu-sm-cores


get-gpu-sm-cores: get-gpu-sm-cores.cu
	nvcc -o $@ $<

clean:
	rm -f get-gpu-sm-cores