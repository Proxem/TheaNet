using System;
using System.Diagnostics;

using static Proxem.TheaNet.Op;

namespace Proxem.TheaNet.Console.Core
{
    class Program
    {
        static void Main(string[] args)
        {
            var x = (Scalar<float>)3.0f;
            var f = Function(output: x);
            Debug.Assert(f() == 3.0f);
        }
    }
}
