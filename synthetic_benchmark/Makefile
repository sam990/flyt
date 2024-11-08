CUSRCS = $(wildcard *.cu)
CSRCS = $(wildcard *.c)
OBJS = $(CUSRCS:.cu=) $(CSRCS:.c=)

all: $(OBJS)
# -cudart=shared : nvcc uses the .so libcudart

%: %.cu
	nvcc $< -O0 -arch sm_86 -Itoml11/single_include -o $@ -lcuda -cudart=shared -lrt -lpthread

%: %.c
	gcc $< -o $@ -lrt -lpthread -ldl

clean:
	rm -f $(OBJS)
