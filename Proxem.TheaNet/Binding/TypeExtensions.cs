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
    public static class TypeExtensions
    {
        public static string GetName(this Type type)
        {
            if (type.IsArray)
            {
                var ilist = type.FindInterfaces((m, criteria) => m.Name == "IList`1", null);
                return ilist[0].GenericTypeArguments[0].GetName() + "[]";
            }
            if (!type.IsGenericType)
            {
                switch (type.FullName)
                {
                    case "System.Int32":
                        return "int";
                    case "System.Single":
                        return "float";
                    case "System.String":
                        return "string";
                    case "System.Double":
                        return "double";
                    case "System.Byte":
                        return "byte";
                    case "System.Char":
                        return "char";
                    default:
                        return type.Name;
                }
            }
            string name = type.GetGenericTypeDefinition().Name;
            name = name.Substring(0, name.IndexOf('`'));
            string args = string.Join(",", type.GetGenericArguments().Select(t => t.GetName()).ToArray());
            return name + "<" + args + ">";
        }
    }
}
