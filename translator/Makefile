# Makefile for ROSE OP2 translator

ROSELIBS:=-L${ROSEDIR}/lib -lrose
ROSEINCLUDE:=-I${ROSEDIR}/include

DEBUG = -g
CXXFLAGS = $(ROSELIBS) $(ROSEINCLUDE) $(DEBUG)

# -I/home/zs1109/install/BOOST_INSTALL/include
#NVCCFLAGS := -arch sm_20
#NVCCFLAGS += -g
#NVCCFLAGS += -Xptxas -v
#NVCCFLAGS += -G

all: op2rose

op2rose: op2rose.cpp rose_op2argument.cpp rose_op2parloop.cpp rose_op2source.cpp rose_op2build.cpp
	g++ $(CXXFLAGS) rose_op2build.cpp rose_op2argument.cpp rose_op2parloop.cpp rose_op2source.cpp op2rose.cpp -o op2rose

clean:
	rm -f op2rose
