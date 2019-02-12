using Microsoft.VisualStudio.TestTools.UnitTesting;
using static Proxem.TheaNet.Op;

namespace Proxem.TheaNet.Test.Core
{
    [TestClass]
    public class UnitTest1
    {
        [TestMethod]
        public void TestScalarConst()
        {
            var x = (Scalar<float>)3.0f;
            var f = Function(output: x);
            Assert.AreEqual(f(), 3.0f);
            Assert.AreNotEqual(f(), 4.0f);
        }
    }
}
