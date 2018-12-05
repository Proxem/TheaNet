/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.

 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Proxem.TheaNet.Binding
{
    [Obsolete]
    public class LoopInvariantCodeMover: CodeGenerator
    {
        public override void VisitVar(IVar var, Compiler compiler)
        {
            // nothing: a variable not yet declared is not an error
            // It indicates that this variable belongs to the loop
            // (it was not declared before processing the loop)
            // and thus that expressions using this variable won't be moved out of the loop
            // e.g. see VisitUnaryElementWise: if (!Compiler.Scope.Contains(unary.x)) return true
        }

        public override bool VisitElementwise<T>(Tensor<T>.Elementwise elementwise, Compiler compiler)
        {
            foreach (var expr in elementwise.Inputs)
            {
                compiler.CompileExpr(expr, this);
            }
            if (!elementwise.Inputs.All(expr => compiler.Scope.Contains(expr))) return true;     // part of the expression was not reachable, exit (processed = true)

            return base.VisitElementwise(elementwise, compiler);
        }
    }
}
