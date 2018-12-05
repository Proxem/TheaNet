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
using Proxem.NumNet;
using Proxem.TheaNet;
using Proxem.TheaNet.Samples.CRF;

using T = Proxem.TheaNet.Op;

namespace Proxem.TheaNet.Test
{
    [TestClass]
    public class TestCrf
    {
        [TestInitialize]
        public void Initialize()
        {
            Runtime.Reset();
        }

        [TestMethod]
        public void TestForwardCrf()
        {
            var rng = NN.Random.Seed(20130601);
            var o = T.Matrix<float>("o");
            var c = T.Tensor3<float>("c");
            var f = T.Function(input1: o, input2: c, output: Crf.Forward(o, c));
            var g = T.Function(input1: o, input2: c, output: Crf.Forward(o, c, viterbi: true));
            for (int i = 0; i < 20; i++)
            {
                var num_labels = rng.Next(2, 10);
                var num_timesteps = rng.Next(2, 10);
                var obs = NN.Random.Uniform(-1, 1, num_timesteps, num_labels);
                var chain = NN.Random.Uniform(-1, 1, num_labels, num_labels, num_labels);
            }
        }
    }
}
