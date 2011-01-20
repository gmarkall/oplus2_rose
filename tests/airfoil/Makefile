OP2ROSE := ../../translator/op2rose
NVCC := nvcc
NVCCFLAGS := -arch sm_13 -deviceemu -g


all: build

build: src2src
	$(NVCC) $(NVCCFLAGS) rose_airfoil.cpp op_par.cu cutil.cpp \
	cmd_arg_reader.cpp stopwatch.cpp bank_checker.cpp \
	stopwatch_linux.cpp -o airfoil

src2src: clean
	$(OP2ROSE) airfoil.cpp

clean:
	rm -f op_par.cu kernels.h res_calc_kernel.cu update_kernel.cu \
	adt_calc_kernel.cu save_soln_kernel.cu rose_airfoil.cpp airfoil \
	blank.cpp test.log

testexec: build
	./airfoil 50 | tee test.log
