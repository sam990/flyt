#include <stdio.h>
#include <unistd.h>
#include <stdlib.h>
#include <dlfcn.h>


int main() {
	void* ret = dlopen("/usr/local/cuda/lib64/libcudarto.so", RTLD_LAZY );
	if (!ret) {
		fprintf(stderr, "Error: %s\n", dlerror());
	}
	return 0;
}


