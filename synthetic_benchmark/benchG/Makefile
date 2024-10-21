# 
# Copyright (c) 2013 No Face Press, LLC
# License http://opensource.org/licenses/mit-license.php MIT License
#

#This makefile is used to test the other Makefiles


PROG = internal-service
SRC = internal-service.c

FUNC_LIB = functions functions_map device_functions
FUNC_LIB_SRC = functions.c functions_map.c device_functions.cu

DEP_HEADERS = functions_map.h

TOP = libhttp-1.8/
CIVETWEB_LIB = libcivetweb.a

CFLAGS = -I$(TOP)/include $(COPT) -g -O0
LIBS = -lpthread

NVCC_FLAGS = -arch sm_86

include $(TOP)/resources/Makefile.in-os

ifeq ($(TARGET_OS),LINUX) 
	LIBS += -ldl
endif

all: $(PROG)

$(PROG): $(CIVETWEB_LIB) $(SRC) $(FUNC_LIB_SRC) $(DEP_HEADERS)
	nvcc -o $@ $(CFLAGS) $(NVCC_FLAGS) $(LDFLAGS) $(SRC) $(FUNC_LIB_SRC) $(CIVETWEB_LIB) $(LIBS) -cudart=shared

$(CIVETWEB_LIB):
	$(MAKE) -C $(TOP) clean lib
	cp $(TOP)/$(CIVETWEB_LIB) .

clean:
	rm -f $(CIVETWEB_LIB) $(PROG) $(FUNC_LIB)

.PHONY: all clean
