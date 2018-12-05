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
using Microsoft.VisualStudio.TestTools.UnitTesting;

using Proxem.LinearAlgebra.Tensors.Single;
using Proxem.Expressions;
using T = Proxem.Expressions.Op;

namespace Proxem.LinearAlgebra.Expressions.Test
{
    [TestClass]
    public class TestSmallNN
    {

        [TestMethod]
        public void Or()
        {
            var trainOR = new TrainingExample(){
                {Tensor.Vector(0f, 0f).T, Tensor.Vector(0f)},
                {Tensor.Vector(0f, 1f).T, Tensor.Vector(1f)},
                {Tensor.Vector(1f, 0f).T, Tensor.Vector(1f)},
                {Tensor.Vector(1f, 1f).T, Tensor.Vector(1f)}
            }.ToArray();

            var net = Network.WithShape(T.Tanh, 2, 1);
            
            var error_target = 0.0001f;
            var error = net.Backprop(0.9f, error_target, 10000, trainOR).Last();
            AssertLessThan(error, error_target);
            
        }

        [TestMethod]
        public void Xor()
        {
            var trainXOR = new TrainingExample(){
                {Tensor.Vector(0f, 0f).T, Tensor.Vector(0f)},
                {Tensor.Vector(0f, 1f).T, Tensor.Vector(1f)},
                {Tensor.Vector(1f, 0f).T, Tensor.Vector(1f)},
                {Tensor.Vector(1f, 1f).T, Tensor.Vector(0f)}
            }.ToArray();

            var net = Network.WithShape(T.Tanh, 2, 2, 1);
            var error_target = 0.05f;
            var error = net.Backprop(0.01f, error_target, 10000, trainXOR).Last();
            AssertLessThan(error, error_target);
        }

        public void AssertLessThan(float a, float b)
        {
            if (!(a < b))
                throw new AssertFailedException(string.Format("AssertLessThan failed. <{0}> is not less than <{1}>", a, b));
        }
    }
}
