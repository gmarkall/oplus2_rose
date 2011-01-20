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

#ifndef ROSE_OP2BUILD_H
#define ROSE_OP2BUILD_H

#include <rose.h>
#include <vector>
#include "rose_op2parloop.h"

using namespace std;
using namespace SageBuilder;
using namespace SageInterface;

struct constVar
{
  int dim;
  SgType* type;
  constVar(int dim, SgType* type) { this->dim = dim; this->type = type; }
};

class OPBuild : public AstSimpleProcessing
{  
  private:
    SgProject *project;
    SgProject *out_project;
    OPParLoop *pl;
    vector<SgDeclarationStatement*> sharedConstVariables;

  public:
    virtual void visit(SgNode *n);
    virtual void atTraversalEnd();

    SgType* getTypeFromExpression(SgExpression* i);
    void setParLoop(OPParLoop *pl);
    void setProject(SgProject *p);
    void generateBuildFile();
    void unparse();
    void setOPParLoop(OPParLoop* pl);
};

#endif
