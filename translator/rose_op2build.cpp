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

#include <iostream>
#include <sstream>

#include "rose_op2build.h"

///////////////////////////// OPBuild : Implementation /////////////////////////

/* 
 *   Build needs to know which ROSE project it is working on
 */
void OPBuild::setProject(SgProject *p)
{
  project = p;
}

/*
 *  Build needs to know about all the kernels
 */
void OPBuild::setParLoop(OPParLoop* pl)
{
  this->pl = pl;
}

/*
 *  Get type from the varrefexp
 */
SgType* OPBuild::getTypeFromExpression(SgExpression* e)
{
  if(isSgAddressOfOp(e))
  {
    return isSgPointerType(isSgAddressOfOp(e)->get_type())->get_base_type();
  }
  return NULL;
}

/*
 *  Replace the op_par_loop with respective kernel function
 */
void OPBuild::generateBuildFile()
{
  // We build a new file for the CUDA kernel and its stub function
  string file_name = "op_par.cu";
  cout << "Generating OP Build File" << endl;
  
  SgSourceFile *file = isSgSourceFile(buildFile("blank.cpp", file_name, NULL));
  ROSE_ASSERT(file!=NULL);
  SgGlobal *globalScope = file->get_globalScope();
  
  // Add important files
  cout << "Adding includes and imports" << endl;
  addTextForUnparser(globalScope, "#include \"op_lib.cu\"\n", AstUnparseAttribute::e_before);
  addTextForUnparser(globalScope, "#import \"op_datatypes.cpp\"\n", AstUnparseAttribute::e_before);
  
  // Add shared const variables
  cout << "Adding shared const variables" << endl;
  for(unsigned int i = 0; i < sharedConstVariables.size() ; ++i)
  {
    SgVariableDeclaration* stmt2 = isSgVariableDeclaration(sharedConstVariables[i]);
    SgVariableDeclaration *varDec = buildVariableDeclaration(
                                      stmt2->get_definition()->get_vardefn()->get_name(), 
                                      stmt2->get_definition()->get_vardefn()->get_type(), 
                                      NULL, 
                                      globalScope);
    addTextForUnparser(varDec, "\n__constant__", AstUnparseAttribute::e_before);
    appendStatement(varDec,globalScope);
  }

  // Add generated kernel files
  map<string, SgFunctionDeclaration*>::iterator p;
  for(p=pl->cudaFunctionDeclarations.begin(); p != pl->cudaFunctionDeclarations.end() ; ++p)
  {
    addTextForUnparser(globalScope, "#include \"" + (*p).first + "_kernel.cu\"\n", AstUnparseAttribute::e_after);
  }
  
  // Add to list of files that need to be unparsed.
   out_project = file->get_project();
  cout << "Finished generating OP Build File" << endl;
}

/*
 * The visit function is reimplemented from AstSimpleProcessing, and is called for every node
 * in the the AST. We are only really interested in looking at function calls to op_par_loop_3.
 */
void OPBuild::visit(SgNode *n)
{
  // We need to put the global scope on the scope stack so that we can look
  // up the oplus datatypes later on (in generateSpecial).
  SgGlobal *globalScope = isSgGlobal(n);
  if(globalScope!=NULL)
  {
    pushScopeStack(globalScope);
  }

  // Lookout for const declaraitons
  SgFunctionCallExp *fn = isSgFunctionCallExp(n);
  if(fn != NULL)
  {
    string fn_name = fn->getAssociatedFunctionDeclaration()->get_name().getString();
    if(fn_name.compare("op_decl_const")==0) 
    {
      SgExpressionPtrList& args = fn->get_args()->get_expressions();

      SgVarRefExp* varRef = isSgVarRefExp(args[1]);
      if(!varRef)
      {
        SgAddressOfOp *e = isSgAddressOfOp(args[1]);
        varRef = isSgVarRefExp(e->get_operand_i());
      }
            
      if(varRef)
      {
        SgDeclarationStatement* decl = varRef->get_symbol()->get_declaration()->get_declaration();
        sharedConstVariables.push_back(decl);
      }
    }
  }
}

/* 
 * Outputs each generated kernel file to disk.
 */
void OPBuild::unparse()
{
  if(out_project)
    out_project->unparse();
  cout << "Finished UNPARSING" << endl;
}

/* 
 * Print out debug info at traversal end
 */
void OPBuild::atTraversalEnd() 
{
  cout << "Traversal Ended." << endl;
}

