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
    public class Scope: IDisposable
    {
        public readonly Scope Parent;
        public readonly int Depth;

        private Dictionary<IExpr, string> Variables = new Dictionary<IExpr, string>();
        private readonly Compiler compiler;

        // TODO: check sooner for colliding names (in IExpr.Name { set;} }
        //private Dictionary<string, int> existingNames = new Dictionary<string, int>();

        public Scope(Compiler compiler) : this(compiler, compiler.Scope) { }

        private Scope(Compiler compiler, Scope parent)
        {
            this.compiler = compiler;
            this.Parent = parent;
            if (parent != null)
            {
                Depth = parent.Depth + 1;

                // TODO: check sooner for colliding names (in IExpr.Name { set;} }
                //foreach (var nameCount in Parent.existingNames)
                //    existingNames.Add(nameCount.Key, nameCount.Value);
            }
        }

        public bool Contains(IExpr e)
        {
            if (this.Variables.ContainsKey(e)) return true;
            if (this.Parent == null) return false;
            return this.Parent.Contains(e);
        }

        public bool Contains(string name)
        {
            if(this.Variables.ContainsValue(name)) return true;
            if (this.Parent == null) return false;
            return this.Parent.Contains(name);
        }


        public void Declare(IConst e, Compiler compiler)
        {
            Variables[e] = e.Name ?? e.Literal;
        }

        public void Declare(ISymbol e, Compiler compiler)
        {
            Variables[e] = e.Name;
        }

        public void Declare(IExpr e, Compiler compiler, string name = null)
        {
            this.Variables[e] = name ?? e.Name ?? "_" + compiler.ID++;
        }

        public string GetVar(IExpr e)
        {
            string result;
            if (this.Variables.TryGetValue(e, out result)) return result;
            if (this.Parent == null)
                throw new KeyNotFoundException($"{e} not found");
            return this.Parent.GetVar(e);
        }

        public void Dispose()
        {
            compiler.Scope = this.Parent;
        }
    }
}
