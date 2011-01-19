# Copyright (c) 2010, Graham Markall
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without modification, 
# are permitted provided that the following conditions are met:
#
#    * Redistributions of source code must retain the above copyright notice, this
#      list of conditions and the following disclaimer.
#    * Redistributions in binary form must reproduce the above copyright notice, 
#      this list of conditions and the following disclaimer in the documentation 
#      and/or other materials provided with the distribution.
#    * Neither the name of Imperial College London nor the names of its 
#      contributors may be used to endorse or promote products derived from this 
#      software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
# ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
# WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR
# ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
# (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON
# ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
# SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#

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
