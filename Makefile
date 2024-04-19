#MIT License...

.PHONY: all cuda-gdb libtirpc gpu cpu tests clean install install-cpu control-managers install-cmgr 

all: cpu install

clean:
	@echo -e "\033[31m----> Cleaning up gpu\033[0m"
	$(MAKE) -C gpu clean
	@echo -e "\033[31m----> Cleaning up cpu\033[0m"
	$(MAKE) -C cpu clean
	@echo -e "\033[31m----> Cleaning up test kernels\033[0m"
	$(MAKE) -C tests clean
	@echo -e "\033[31m----> Removing bin...\033[0m"
	rm -rf bin
	@echo -e "\033[31m All done!\033[0m"

cuda-gdb:
	@echo -e "\033[36m----> Building submodules\033[0m"
	$(MAKE) -C submodules cuda-gdb
	$(MAKE) -C submodules cuda-gdb-libs

libtirpc:
	@echo -e "\033[36m----> Building libtirpc\033[0m"
	$(MAKE) -C submodules libtirpc/install

gpu: cuda-gdb
	@echo -e "\033[36m----> Building gpu\033[0m"
	$(MAKE) -C gpu

cpu: libtirpc
	@echo -e "\033[36m----> Building cpu\033[0m"
	$(MAKE) -C cpu

cpu-client: libtirpc
	@echo -e "\033[36m----> Building cpu-client\033[0m"
	$(MAKE) -C cpu cricket-client.so

install-cpu-client: cpu-client bin/cricket-client.so
	@echo -e "\033[36m----> Copying cpu-client to build/bin\033[0m"

control-managers:
	@echo -e "\033[36m----> Building control-managers\033[0m"
	$(MAKE) -C control-managers

tests:
	@echo -e "\033[36m----> Building test kernels\033[0m"
	$(MAKE) -C tests

install-cpu: bin/cricket-client.so bin/cricket-rpc-server bin/libtirpc.so bin/libtirpc.so.3
	@echo -e "\033[36m----> Copying cpu binaries to build/bin\033[0m"

install-tests: bin/tests
	@echo -e "\033[36m----> Copying test binaries to build/bin\033[0m"

install-gpu: bin/cricket
	@echo -e "\033[36m----> Copying gpu binaries to build/bin\033[0m"

install-cmgr: control-managers
	$(MAKE) -C control-managers install

install: install-cpu install-cmgr install-tests
	@echo -e "\033[36m----> Copying to build/bin\033[0m"

bin:
	mkdir bin

bin/tests: bin tests
	ln -sf ../tests/bin bin/tests

bin/cricket-client.so: bin cpu
	cp cpu/cricket-client.so bin

bin/cricket-server.so: bin
	$(MAKE) -C cpu cricket-server.so
	cp cpu/cricket-server.so bin/cricket-server.so


bin/cricket-rpc-server: bin cpu
	cp cpu/cricket-rpc-server bin/cricket-rpc-server

bin/cricket: bin gpu
	cp gpu/cricket bin

bin/libtirpc.so: bin submodules/libtirpc/install/lib/libtirpc.so
	cp submodules/libtirpc/install/lib/libtirpc.so bin

bin/libtirpc.so.3: bin submodules/libtirpc/install/lib/libtirpc.so.3 libtirpc
	cp submodules/libtirpc/install/lib/libtirpc.so.3 bin
