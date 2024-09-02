CUSRCS = $(wildcard *.cu)
CSRCS = $(wildcard *.c)
OBJS = $(CUSRCS:.cu=) $(CSRCS:.c=)

all: $(OBJS)
# -cudart=shared : nvcc uses the .so libcudart

%: %.cu
	nvcc $< -O0 -arch sm_86 -o $@ -lcuda -cudart=shared -lrt -lpthread -ldl

%: %.c
	gcc $< -o $@ -lrt -lpthread -ldl

clean:
	rm -f $(OBJS)
