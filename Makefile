#MIT License...

TARGET_LIBCUDART_MAJOR_VERS = 12
export TARGET_LIBCUDART_MAJOR_VERS

.PHONY: all cuda-gdb libtirpc gpu cpu clean install install-cpu control-managers # tests
.PHONY: install-cmgr install-cpu-server install-cpu-client install-tests install-gpu

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


cpu-server: libtirpc
	@echo -e "\033[36m----> Building cpu-server\033[0m"
	$(MAKE) -C cpu cricket-rpc-server

cpu-client: libtirpc
	@echo -e "\033[36m----> Building cpu-client\033[0m"
	$(MAKE) -C cpu cricket-client.so

cpu: libtirpc cpu-server cpu-client
	@echo -e "\033[36m----> Building cpu\033[0m"

install-cpu-client: cpu-client bin/cricket-client.so
	@echo -e "\033[36m----> Copying cpu-client to build/bin\033[0m"

control-managers:
	@echo -e "\033[36m----> Building control-managers\033[0m"
	$(MAKE) -C control-managers

# tests:
# 	@echo -e "\033[36m----> Building test kernels\033[0m"
# 	$(MAKE) -C tests

install-cpu-server: bin/cricket-rpc-server
	@echo -e "\033[36m----> Copying cpu-server to build/bin\033[0m"

install-cpu: install-cpu-server install-cpu-client bin/libtirpc.so bin/libtirpc.so.3
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

bin/cricket-client.so: bin cpu-client
	cp cpu/cricket-client.so bin

bin/cricket-server.so: bin cpu-server
	$(MAKE) -C cpu cricket-server.so
	cp cpu/cricket-server.so bin/cricket-server.so

bin/cricket-rpc-server: bin cpu-server
	cp cpu/cricket-rpc-server bin/cricket-rpc-server

bin/cricket: bin gpu
	cp gpu/cricket bin

bin/libtirpc.so: bin submodules/libtirpc/install/lib/libtirpc.so
	cp submodules/libtirpc/install/lib/libtirpc.so bin

bin/libtirpc.so.3: bin submodules/libtirpc/install/lib/libtirpc.so.3 libtirpc
	cp submodules/libtirpc/install/lib/libtirpc.so.3 bin

# Copy the cricket-client.so contents (i.e. rpc calls) to the original cuda runtime library .so
# symbolic link the original rt.so library to the newly created .999 library (containing cricket client implementation)
# Once installed, regular cuda calls will be implemented by this new library.
install-client-lib: bin/cricket-client.so /usr/local/cuda/lib64/libcudarto.so
	@echo -e "\033[36m----> Installing vcuda library\033[0m"
	sudo cp bin/cricket-client.so /usr/local/cuda/lib64/libcudart.so.$(TARGET_LIBCUDART_MAJOR_VERS).9.999
	sudo sh -c "cd /usr/local/cuda/lib64 && ln -sf libcudart.so.$(TARGET_LIBCUDART_MAJOR_VERS).9.999 libcudart.so.$(TARGET_LIBCUDART_MAJOR_VERS) && ldconfig"
	sudo ln -s /usr/local/cuda/lib64/libcudart.so /usr/lib/x86_64-linux-gnu/libcuda.so.1

restore-client-lib: /usr/local/cuda/bkp
	@echo -e "\033[36m----> Restoring original library\033[0m"
	sudo cp -f /usr/local/cuda/bkp/* /usr/local/cuda/lib64/
	sudo rm -rf /usr/local/cuda/bkp
	sudo rm -rf /usr/local/cuda/lib64/libcudart.so.$(TARGET_LIBCUDART_MAJOR_VERS).9.999
	sudo rm -rf /usr/local/cuda/lib64/libcudarto.so*
	sudo rm /usr/lib/x86_64-linux-gnu/libcuda.so.1
	sudo sh -c "cd /usr/local/cuda/lib64 && ldconfig"

submodules/patchelf/install/bin/patchelf:
	@echo -e "\033[36m----> Building patchelf\033[0m"
	$(MAKE) -C submodules patchelf/install

/usr/local/cuda/bkp:
	@echo -e "\033[36m----> Backup of original library\033[0m"
	sudo mkdir -p /usr/local/cuda/bkp
	$(eval CUDA_LIB := $(shell readlink -f /usr/local/cuda/lib64/libcudart.so))
	sudo cp $(CUDA_LIB) /usr/local/cuda/bkp

/usr/local/cuda/lib64/libcudarto.so: submodules/patchelf/install/bin/patchelf /usr/local/cuda/bkp
	@echo -e "\033[36m----> Patching libcudart.so\033[0m"
	$(eval CUDA_LIB := $(shell readlink -f /usr/local/cuda/lib64/libcudart.so))
	$(eval LIB_VERSION := $(shell basename ${CUDA_LIB} | sed 's/libcudart\.so\.//g'))
	sudo cp ${CUDA_LIB} /usr/local/cuda/lib64/libcudarto.so.${LIB_VERSION}
	sudo submodules/patchelf/install/bin/patchelf --set-soname libcudarto.so.$(TARGET_LIBCUDART_MAJOR_VERS) /usr/local/cuda/lib64/libcudarto.so.$(LIB_VERSION)
	sudo ln -sf /usr/local/cuda/lib64/libcudarto.so.$(LIB_VERSION) /usr/local/cuda/lib64/libcudarto.so.$(TARGET_LIBCUDART_MAJOR_VERS)
	sudo ln -sf /usr/local/cuda/lib64/libcudarto.so.$(TARGET_LIBCUDART_MAJOR_VERS) /usr/local/cuda/lib64/libcudarto.so
	sudo sh -c "cd /usr/local/cuda/lib64 && ldconfig"
