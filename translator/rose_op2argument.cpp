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

#include "rose_op2argument.h"

const int op_argument::num_params = 4;

// GM Note - not sure what this comment means right now.
// Only one way has been handled
// SgAssignInitializer---SgVarRefExp (Implemented)
// SgAssignInitializer---SgAddressOfOp---SgVarRefExp(TODO)
// SgConstructorInitializer---SgVarRefExp (TODO)
SgVarRefExp* op_argument::getSgVarRefExp(SgExpression* i)
{
  SgVarRefExp* ref = NULL;
  SgAddressOfOp* op = isSgAddressOfOp(i);
  if(op)
  {
    ref = isSgVarRefExp(op->get_operand_i());
  }
  SgCastExp* cast = isSgCastExp(i);
  if(cast)
  {
    SgAddressOfOp* op = isSgAddressOfOp(cast->get_operand_i());
    if(op)
    {
      ref = isSgVarRefExp(op->get_operand_i());
    }
  }
  
  return ref;
}

// Returns the SgType of the op_dat - float, double or int
SgType* op_argument::getSgTypeFromVarRef(SgVarRefExp* arg)
{
  SgConstructorInitializer* initer = isSgConstructorInitializer(arg->get_symbol()->get_declaration()->get_initptr());

  SgTemplateInstantiationDecl* template_decl = isSgTemplateInstantiationDecl(initer->get_class_decl());
  if(template_decl)
  {
    SgTemplateArgumentPtrList args = template_decl->get_templateArguments();
    SgTemplateArgumentPtrList::iterator r = args.begin();
    SgTemplateArgument* expr = *r;
    return expr->get_type();
  }
  return NULL;
}

// Returns the Dim of the op_dat - 1, 2, 4 etc
int op_argument::getDimFromVarRef(SgVarRefExp* arg, bool global)
{
  SgConstructorInitializer* initer = isSgConstructorInitializer(arg->get_symbol()->get_declaration()->get_initptr());
  SgExpressionPtrList& args = initer->get_args()->get_expressions();
  
  // Gets passed in as the 1st argument for category=global, 2nd argument for category=non_global
  SgExpressionPtrList::iterator q = args.begin();  
  if(!global)
    q++;

  return isSgIntVal(*q)->get_value();
}

bool op_argument::consideredAsReduction()
{
  return global && (access != OP_READ);
}

bool op_argument::isGlobal()
{
  return global;
}

bool op_argument::isNotGlobal()
{
  return !global;
}

bool op_argument::consideredAsConst()
{
  return global && (access == OP_READ);
}

// Constructor
op_argument::op_argument(SgExpressionPtrList::iterator &i)
{
  global = false;
  arg = getSgVarRefExp(*i++);
  index = isSgIntVal(*i++)->get_value();
  
  // Checking for NULL
  SgExpression* exp = *i++;
  SgCastExp* cast_exp = isSgCastExp(exp);
  if(cast_exp)
    ptr = cast_exp->get_originalExpressionTree();
  else
    ptr = exp;
  
  //dim = isSgIntVal(*i++)->get_value();  // We fetch dim based on the category
  access = (op_access)(isSgEnumVal(*i++)->get_value()); // might have bug if differs from the ones used by the compiler.
  type = getSgTypeFromVarRef(arg);
  plan_index = -1;

  // identify category
  SgType* t = arg->get_symbol()->get_declaration()->get_typeptr();
  SgNamedType* type = isSgNamedType(t);
  if(type)
  {
    string name = type->get_name().getString();
    size_t found = name.find("op_dat_gbl");
    if( found!=string::npos )
    {
      global = true;
    }
  }
  
  // fetch dimension
  dim = getDimFromVarRef(arg, global);
}

// Accessor
bool op_argument::usesIndirection()
{
  return  !(ptr == NULL);
}
