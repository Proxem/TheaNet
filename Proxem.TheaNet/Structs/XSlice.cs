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

using Proxem.NumNet;

using Int = Proxem.TheaNet.Scalar<int>.Const;
using Axis = Proxem.TheaNet.Scalar<int>;

namespace Proxem.TheaNet
{
    /// <summary>
    /// Expression representing a NumNet Slice.
    /// Can be implicitly converted from a NumNet Slice, or a integer.
    /// </summary>
    public class XSlice: Expr, IExpr<Slice>
    {
        public static XSlice Create(Scalar<int> start, Scalar<int> stop) => Create(start, stop, 1);

        public static XSlice Create(Scalar<int> start, Scalar<int> stop, Scalar<int> step)
        {
            if (stop == null) System.Diagnostics.Debugger.Break();
            if (step.IsZero)
                return new XSlice(start);
            else
                return new XSlice(start, stop, step);
        }

        private XSlice(Scalar<int> start, Scalar<int> stop, Scalar<int> step): base("Slice", start, stop, step)
        {
            IsSingleton = false;
        }

        private XSlice(Scalar<int> start) : base("Slice", start)
        {
            IsSingleton = true;
        }

        public Scalar<int> Start => (Scalar<int>)Inputs[0];

        public Scalar<int> Stop => (Scalar<int>)Inputs[1];

        public Scalar<int> Step => (Scalar<int>)Inputs[2];

        public static IReadOnlyList<XSlice> Patch(Patch substitutions, IReadOnlyList<XSlice> slices)
        {
            var ok = true;
            var result = new XSlice[slices.Count];
            for (int i = 0; i < result.Length; i++)
            {
                var slice = slices[i];
                result[i] = new XSlice(
                    (Scalar<int>)slice.Start.Patch(substitutions),
                    (Scalar<int>)slice.Stop.Patch(substitutions),
                    (Scalar<int>)slice.Step.Patch(substitutions)
                );
                if (result[i].Start != slice.Start ||
                    result[i].Stop != slice.Stop ||
                    result[i].Step != slice.Step)
                    ok = false;
            }
            if (ok) return slices;
            return result;
        }

        public bool IsSingleton { get; }

        /// <summary>
        /// Returns the size of this slice, 1 for a singleton.
        /// </summary>
        public Axis Size()
        {
            if (IsSingleton) return 1;
            else return (Stop - Start) / Step;
        }

        public static implicit operator XSlice(int i) => new XSlice(i);

        public static implicit operator XSlice(Scalar<int> i) => new XSlice(i);

        public static implicit operator XSlice(Slice s) => Create(s.Start, s.Stop, s.Step);

        public override string ToString()
        {
            var start = Start.ToString();
            if (this.IsSingleton) return start;

            var result = new StringBuilder();
            if (start != "0") result.Append(start);
            result.Append(':');
            var stop = Stop.ToString();
            if (stop != int.MaxValue.ToString() && stop != int.MinValue.ToString()) result.Append(this.Stop);
            if (this.Step.ToString() != "1")
            {
                result.Append(':');
                result.Append(this.Step);
            }
            return result.ToString();
        }

        public Axis ExtractAxis<T>(Tensor<T> t, int i)
        {
            if (IsSingleton)
                return null;

            var axis = t.Shape[i];
            if (Step.IsOne)
            {
                // [x:]
                if (Stop == null || Stop.IsConst(int.MaxValue))
                    return axis - GetAbsoluteIndex(t, i, Start);
            }
            else if (Step.Check((Int s) => s.Value < 0))
            {
                // [x::-s]
                if (Stop.IsConst(int.MinValue))
                    // [::-s]
                    if (Start.IsMinusOne)
                        return axis / -Step;
                    else
                        return (GetAbsoluteIndex(t, i, Start) + 1) / -Step;
            }

            var stop = GetAbsoluteIndex(t, i, Stop);
            var start = GetAbsoluteIndex(t, i, Start);

            return (stop - start) / Step;
        }

        private static Scalar<int> GetAbsoluteIndex<T>(Tensor<T> t, int i, Scalar<int> axis)
        {
            switch (axis)
            {
                case Int s:
                    return GetAbsoluteIndex(t, i, s.Value);
                default:
                    return axis;
            }
        }

        private static Scalar<int> GetAbsoluteIndex<T>(Tensor<T> t, int i, int index)
        {
            var axis = t.Shape[i];
            if (index == int.MaxValue)
                return axis;
            //else if (index == int.MinValue)
            //    return -1;
            else if(index < 0)
                return axis + index;
            else
                return index;
        }

        public override void Process(IProcessor processor) => processor.ProcessSlice(this);

        public override IExpr Patch(Patch substitutions)
        {
            XSlice result;
            if (substitutions.TryGetValue(this, out result)) return result;
            var inputsPatched = Inputs.Patch(substitutions);

            if (inputsPatched != Inputs)
            {
                XSlice patched;
                if (IsSingleton)
                    patched = new XSlice((Scalar<int>)inputsPatched[0]);
                else
                    patched = new XSlice((Scalar<int>)inputsPatched[0], (Scalar<int>)inputsPatched[1], (Scalar<int>)inputsPatched[2]);
                substitutions.Add(this, patched);
                return patched;
            }
            else
                return this;
        }

        public override Type GetArgumentType() => typeof(Slice);
    }

    /// <summary>
    /// Static class to simplify XSlice creation. Similar to NumNet.Slicer.
    /// </summary>
    public static class XSlicer
    {
        public static readonly XSlice _ = Range(0, null);
        //public static readonly Slice NewAxis = Range(0, 0, int.MaxValue);

        public static XSlice Range(Scalar<int> start, Scalar<int> stop, int step = 1)
        {
            if (stop == null) return From(start, step);
            if (start == null) return Until(stop, step);
            return XSlice.Create(start, stop, step);
        }

        public static XSlice Only(Scalar<int> i) => XSlice.Create(i, i + 1, 0);

        public static XSlice Step(int step)
        {
            if (step < 0) return XSlice.Create(-1, int.MinValue, step);
            return XSlice.Create(0, int.MaxValue, step);
        }

        public static XSlice Until(Scalar<int> stop, int step = 1)
        {
            if (step < 0) return XSlice.Create(-1, stop, step);
            else return XSlice.Create(0, stop, step);
        }

        public static XSlice From(Scalar<int> start, int step = 1)
        {
            if (step < 0) return XSlice.Create(start, int.MinValue, step);
            else return XSlice.Create(start, int.MaxValue, step);
        }
    }
}
