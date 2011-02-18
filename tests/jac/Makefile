COMMONDIR := ../common
OP2ROSE := ../../translator/op2rose

NVCC := nvcc
INCLUDEFLAGS := -I$(COMMONDIR)
NVCCFLAGS := -arch sm_13 -g $(INCLUDEFLAGS)

VPATH = $(COMMONDIR)

SOURCES = rose_jac.cpp op_par.cu cutil.cpp \
	cmd_arg_reader.cpp stopwatch.cpp bank_checker.cpp stopwatch_linux.cpp \

all: src2src build

build: $(SOURCES)
	$(NVCC) $(NVCCFLAGS) $^ -o jac

src2src: clean
	$(OP2ROSE) $(INCLUDEFLAGS) jac.cpp

clean: 
	rm -f op_par.cu kernels.h res_kernel.cu update_kernel.cu rose_jac.cpp \
	jac blank.cpp test.log

testexec: build
	./jac | tee test.log
