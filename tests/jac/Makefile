OP2ROSE := ../../translator/op2rose
NVCC := nvcc
NVCCFLAGS := -arch sm_13 -deviceemu


all: build

build: src2src
	$(NVCC) $(NVCCFLAGS) rose_jac.cpp op_par.cu cutil.cpp \
	cmd_arg_reader.cpp stopwatch.cpp bank_checker.cpp stopwatch_linux.cpp \
	-o jac

src2src: clean
	$(OP2ROSE) jac.cpp

clean: 
	rm -f op_par.cu kernels.h res_kernel.cu update_kernel.cu rose_jac.cpp \
	jac blank.cpp test.log

testexec: build
	./jac | tee test.log
