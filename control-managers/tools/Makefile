all: get-gpu-sm-cores


get-gpu-sm-cores: get-gpu-sm-cores.cu
	nvcc -o $@ $<

install: ../../bin/get-gpu-sm-cores
	
../../bin/get-gpu-sm-cores: get-gpu-sm-cores
	@cp -v $< $@

clean:
	rm -f get-gpu-sm-cores