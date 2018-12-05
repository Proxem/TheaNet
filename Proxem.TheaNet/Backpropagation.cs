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
using System.Diagnostics;
using System.Linq;

namespace Proxem.TheaNet
{
    public class Backpropagation
    {
        public Dictionary<IScalar, IScalar> _scalarDerivatives = new Dictionary<IScalar, IScalar>();
        private Dictionary<ITensor, ITensor> _tensorDerivatives = new Dictionary<ITensor, ITensor>();
        private Dictionary<IExpr, int> deltaReceived = new Dictionary<IExpr, int>();

        public IReadOnlyDictionary<IScalar, IScalar> ScalarDerivatives => _scalarDerivatives;
        public IReadOnlyDictionary<ITensor, ITensor> TensorDerivatives => _tensorDerivatives;

        private LinkedList<IExpr> nodeQueue = new LinkedList<IExpr>();
        private HashSet<IExpr> visited = new HashSet<IExpr>();

        bool firstPass = true; bool incrementTargetInputs = true;
        int needSecondPass = 0;

        public static Backpropagation Backward<T>(Scalar<T> expr, Scalar<T> delta)
        {
            var bp = new Backpropagation();
            bp.Backpropagate(expr, delta);
            return bp;
        }

        public static Backpropagation Backward<T>(Tensor<T> expr, Tensor<T> delta)
        {
            var bp = new Backpropagation();
            delta.AssertOfShape(expr);
            bp.Backpropagate(expr, delta);
            return bp;
        }

        public void Backpropagate<T>(Scalar<T> expr, Scalar<T> delta)
        {
            PushGradientTo(expr, delta);

            if (needSecondPass > 0)
            {
                Trace.WriteLine($"Found {needSecondPass} duplicated deltas, will do another pass.");

                Clear();
                firstPass = false;

                PushGradientTo(expr, delta);
                while (nodeQueue.Count > 0)
                    ProcessQueue();

                Trace.WriteLine($"Second pass done, there shouldn't be any more duplicated deltas");
            }
            else
                Trace.WriteLine("No duplicated delta found");
        }

        public void Backpropagate<T>(Tensor<T> expr, Tensor<T> delta)
        {
            PushGradientTo(expr, delta);

            if (needSecondPass > 0)
            {
                Trace.WriteLine($"Found {needSecondPass} duplicated deltas, will do another pass.");

                Clear();
                firstPass = false;

                PushGradientTo(expr, delta);
                while (nodeQueue.Count > 0)
                    ProcessQueue();

                Trace.WriteLine($"Second pass done, there shouldn't be any more duplicated deltas");
            }
            else
                Trace.WriteLine("No duplicated delta found");
        }

        void Clear()
        {
            visited.Clear();
            nodeQueue.Clear();
            _tensorDerivatives.Clear();
            _scalarDerivatives.Clear();
            needSecondPass = 0;
        }

        public void PushGradientTo<T>(Scalar<T> target, Scalar<T> delta)
        {
            if (delta.IsZero) return;

            if (!_scalarDerivatives.ContainsKey(target))
                _scalarDerivatives[target] = delta;
            else
                _scalarDerivatives[target] = (Scalar<T>)_scalarDerivatives[target] + delta;

            if (!firstPass)
            {
                if (!nodeQueue.Contains(target))
                    nodeQueue.AddFirst(target);
                deltaReceived[target] -= 1;
            }
            else
            {
                if (incrementTargetInputs)
                {
                    if (!deltaReceived.ContainsKey(target))
                        deltaReceived[target] = 1;
                    else
                        deltaReceived[target] += 1;
                }

                var duplicated = visited.Contains(target);
                if (duplicated)
                    needSecondPass += 1;

                var copy = incrementTargetInputs;
                incrementTargetInputs = incrementTargetInputs && !duplicated;
                visited.Add(target);

                target.Backward(delta, this);
                incrementTargetInputs = copy;
            }
        }

        public void PushGradientTo<T>(Tensor<T> target, Tensor<T> delta)
        {
            delta.AssertOfShape(target);
            if (delta.IsZero) return;

            if (!_tensorDerivatives.ContainsKey(target))
                _tensorDerivatives[target] = delta;
            else
                _tensorDerivatives[target] = (Tensor<T>)_tensorDerivatives[target] + delta;

            if (!firstPass)
            {
                if (!nodeQueue.Contains(target))
                    nodeQueue.AddFirst(target);
                deltaReceived[target] -= 1;
            }
            else
            {
                if (incrementTargetInputs)
                {
                    if (!deltaReceived.ContainsKey(target))
                        deltaReceived[target] = 1;
                    else
                        deltaReceived[target] += 1;
                }

                var duplicated = visited.Contains(target);
                if (duplicated && target.Inputs != null && target.Inputs.Count() > 0)
                    needSecondPass += 1;

                var copy = incrementTargetInputs;
                incrementTargetInputs = incrementTargetInputs && !duplicated;
                visited.Add(target);

                target.Backward(delta, this);
                incrementTargetInputs = copy;
            }
        }

        void ProcessQueue()
        {
            var target = nodeQueue.FirstOrDefault(t => deltaReceived[t] == 0);

            if (target == null)
                throw new Exception("Looping delta problem");

            if (visited.Contains(target))
                throw new Exception($"Already visited {target}");

            visited.Add(target);
            dynamic x = target;
            BackwardBreadth_(x);
            nodeQueue.Remove(target);
        }

        void BackwardBreadth_<T>(Scalar<T> target)
        {
            var delta = (Scalar<T>)_scalarDerivatives[target];
            if (delta == null) return;

            target.Backward(delta, this);
        }

        void BackwardBreadth_<T>(Tensor<T> target)
        {
            var delta = (Tensor<T>)_tensorDerivatives[target];
            if (delta == null) return;

            target.Backward(delta, this);
        }
    }
}
