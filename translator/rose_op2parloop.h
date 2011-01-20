/*
 * 
 * Copyright (c) 2010, Graham Markall
 * All rights reserved.
 * 
 * Redistribution and use in source and binary forms, with or without modification,
 * are permitted provided that the following conditions are met:
 * 
 *     * Redistributions of source code must retain the above copyright notice,
 *       this list of conditions and the following disclaimer.
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

#ifndef ROSE_OP2PARLOOP_H
#define ROSE_OP2PARLOOP_H

#include <rose.h>
#include <vector>
#include <map>
#include <string>

#include "rose_op2argument.h"

using namespace std;
using namespace SageBuilder;
using namespace SageInterface;

class op_par_loop
{
  public:
    static const int num_params;
    string prev_name;

    SgFunctionRefExp *kernel;
    SgVarRefExp *set;
    // All args
    vector<op_argument*> args;          
    // Args that do uses indirection
    vector<op_argument*> ind_args;      
    // Arguments are passed into the loop in such a way so that args that uses
    // same op_dat reside together.  The planContainer stores the starting argument
    // of a op_dat category. The size of the container will therefore give us
    // the number of category there are. This neglects all arguments that do not
    // use indirection.
    vector<op_argument*> planContainer;  

    int numArgs() { return args.size(); }
    int numIndArgs() { return ind_args.size(); }

    void updatePlanContainer(op_argument* arg);
};

// The ParLoop class inherits from AstSimpleProcessing, which provides a mechanism for 
// traversing the AST of the program in ROSE. We use this for convenience, as opposed to
// trying to walk through the AST "manually".

class OPParLoop : public AstSimpleProcessing 
{
  private:
    SgProject *project;
    vector<SgProject*> kernels;

  public:
    virtual void visit(SgNode *n);
    virtual void atTraversalEnd();
    
    void generateSpecial(SgFunctionCallExp *fn, op_par_loop *pl);
    void generateStandard(SgFunctionCallExp *fn, op_par_loop *pl);
    void generateGlobalKernelsHeader();
    inline string getName(SgFunctionRefExp *fn);
    void setProject(SgProject *p);
    void unparse();


    //void forwardDeclareUtilFunctions(SgGlobal* globalScope, SgType* op_set, SgType* op_dat, SgType* op_ptr, SgType* op_access, SgType* op_plan);
    void preHandleConstAndGlobalData(SgFunctionCallExp *fn, op_par_loop *pl, SgBasicBlock *body);
    void postHandleConstAndGlobalData(SgFunctionCallExp *fn, op_par_loop *pl, SgBasicBlock *body);
    void preKernelGlobalDataHandling(SgFunctionCallExp *fn, op_par_loop *pl, SgBasicBlock *body);  
    void postKernelGlobalDataHandling(SgFunctionCallExp *fn, op_par_loop *pl, SgBasicBlock *body);


    map<string, SgFunctionDeclaration*> cudaFunctionDeclarations;
};

#endif
