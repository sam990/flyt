
CMGR_BINARIES = flyt-client-manager flyt-node-manager flyt-cluster-manager flytctl flytctlnet

.PHONY: all cargo-build install clean

all: cargo-build

cargo-build:
	cargo build --release

install:
	@echo "\033[36m----> Copying control-manager binaries to bin\033[0m"
	@mkdir -p ../bin
	@for binary in $(CMGR_BINARIES); do \
		cp -v target/release/$$binary ../bin; \
	done

clean:
	cargo clean --release