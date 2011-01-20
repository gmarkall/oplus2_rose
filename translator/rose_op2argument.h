/*
 * 
 * Copyright (c) 2010, Graham Markall
 * All rights reserved.
 * 
 * Redistribution and use in source and binary forms, with or without modification,
 * are permitted provided that the following conditions are met:
 * 
 *     * Redistributions of source code must retain the above copyright notice,
 *      this list of conditions and the following disclaimer.
 *     * Redistributions in binary form must reproduce the above copyright notice,
 *       this list of conditions and the following disclaimer in the documentation
 *       and/or other materials provided with the distribution.
 *     * Neither the name of Imperial College London nor the names of its
 *       contributors may be used to endorse or promote products derived from this
 *       software without specific prior written permission.
 * 
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR
 * ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON
 * ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 * 
 */

#ifndef ROSE_OP2ARGUMENT_H
#define ROSE_OP2ARGUMENT_H

#include <rose.h>

using namespace std;
using namespace SageBuilder;
using namespace SageInterface;

// Enums and classes for storing information about the arguments to
// op_par_loop_* internally

enum op_access   { OP_READ, OP_WRITE, OP_RW, OP_INC, OP_MIN, OP_MAX };

class op_argument
{
  public:
    static const int num_params;
  
    SgVarRefExp *arg;
    int index;
    SgExpression *ptr;
    int dim;
    SgType* type;
    op_access access;
    bool global;
    int plan_index;
    int own_index;
     
    bool usesIndirection();
    bool consideredAsReduction();
    bool isGlobal();
    bool isNotGlobal();
    bool consideredAsConst();

    op_argument(SgExpressionPtrList::iterator &i);
  
  protected:
    SgVarRefExp* getSgVarRefExp(SgExpression* i);
    SgType* getSgTypeFromVarRef(SgVarRefExp* arg);
		int getDimFromVarRef(SgVarRefExp* arg, bool isGlobal);
};

#endif
