
CMGR_BINARIES = flyt-client-manager flyt-node-manager flyt-resource-manager rmgr-cli

.PHONY: all cargo-build install clean

all: cargo-build dep-tools

cargo-build:
	cargo build --release

dep-tools:
	$(MAKE) -C tools

install:
	@echo "\033[36m----> Copying control-manager binaries to bin\033[0m"
	@mkdir -p ../bin
	@for binary in $(CMGR_BINARIES); do \
		cp -v target/release/$$binary ../bin; \
	done
	$(MAKE) -C tools install

clean:
	cargo clean --release