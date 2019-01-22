using System;
using System.Collections.Generic;
using System.Text;
using Proxem.NumNet;
using Proxem.NumNet.Single;

namespace Proxem.TheaNet.Samples
{
    class Score
    {
        // f1 score
        // matrice de confusion

        private static float Precision(Array<float> confMatrix, int positiveLabel = 1)
        {
            float total = confMatrix[positiveLabel, Slicer._].Sum();
            return (float)confMatrix[positiveLabel, positiveLabel] / (float)total;
        }

        private static float Recall(Array<float> confMatrix, int positiveLabel = 1)
        {
            float total = confMatrix[Slicer._, positiveLabel].Sum();
            return (float)confMatrix[positiveLabel, positiveLabel] / (float)total;
        }

        private static float F1Score(Array<float> confMatrix)
        {
            float precision = Precision(confMatrix);
            float recall = Recall(confMatrix);
            return 2 * precision * recall / (precision + recall);
        }

        public static float F1Score(int[] predictions, int[] labels, bool verbose = false)
        {
            Array<float> confMatrix = ConfusionMatrix(predictions, labels, 2);
            if (verbose)
            {
                Console.WriteLine(confMatrix);
            }
            return F1Score(confMatrix);
        }

        public static Array<float> ConfusionMatrix(int[] predictions, int[] labels, int nbLabels)
        {
            Array<float> confMatrix = new Array<float>(nbLabels, nbLabels);
            for (int index = 0; index < predictions.Length; index++)
            {
                int predlabel = predictions[index];
                int truelabel = labels[index];
                confMatrix[predlabel, truelabel] += 1;
            }
            return confMatrix;
        }
    }
}
