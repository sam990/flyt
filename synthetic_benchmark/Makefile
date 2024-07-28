CUSRCS = $(wildcard *.cu)
CSRCS = $(wildcard *.c)
OBJS = $(CUSRCS:.cu=) $(CSRCS:.c=)

all: $(OBJS)

%: %.cu
	nvcc $< -O0 -arch sm_86 -o $@ -lcuda -cudart=shared -lrt -lpthread

%: %.c
	gcc $< -o $@ -lrt -lpthread

clean:
	rm -f $(OBJS)
