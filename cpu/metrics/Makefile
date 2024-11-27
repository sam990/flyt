PROFILER_HOST_UTILS_SRC := extensions
CUDA_SRC = /usr/local/cuda

CC := gcc -g
AR := ar
LIBEXT := so
LIBPREFIX := lib
INCLUDES := -I$(CUDA_SRC)/include -Iinclude -Iinclude/profilerhost_util -Iinclude/c_util

LIBS += -L$(CUDA_SRC)/lib64 -lcuda #-lcupti -lnvperf_host -lnvperf_target

# Source and Object Files
METRICS_SRC = $(wildcard $(PROFILER_HOST_UTILS_SRC)/*.cpp)
OBJ_FILES = $(METRICS_SRC:%.cpp=%.o)

# Default target
.DEFAULT: all
.PHONY: all clean

all: $(LIBPREFIX)profiler.$(LIBEXT)

$(LIBPREFIX)profiler.$(LIBEXT): profiler.o $(OBJ_FILES)
	$(AR) rcs $@ $^

$(PROFILER_HOST_UTILS_SRC)/%.o: $(PROFILER_HOST_UTILS_SRC)/%.cpp
	$(CC) -c -fpic $(INCLUDES) $< -o $@

profiler.o: profiler.cpp
	$(CC) -c -fpic $(INCLUDES) profiler.cpp -o $@

clean:
	rm -f $(LIBPREFIX)profiler.$(LIBEXT) profiler.o $(OBJ_FILES)

