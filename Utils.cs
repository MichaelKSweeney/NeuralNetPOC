using Newtonsoft.Json;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeuralNetPOC
{
    public class Utils
    {
        public static (double[,] trainingSet, double[,] evaluationSet) SplitDataset(double[,] data, double trainingRatio)
        {
            if (trainingRatio > 1) { throw new Exception("Training Data is above the size of the data"); }
            if (trainingRatio <= 0) { throw new Exception("No ratio of data is allocated"); }

            int totalRows = data.GetLength(0);
            int totalColumns = data.GetLength(1);
            // Calculate the split index
            int trainingRows = (int)(totalRows * trainingRatio);

            // Initialize training and evaluation sets
            double[,] trainingSet = new double[trainingRows, totalColumns];
            double[,] evaluationSet = new double[totalRows - trainingRows, totalColumns];

            // Fill training set
            for (int i = 0; i < trainingRows; i++)
            {
                for (int j = 0; j < totalColumns; j++)
                {
                    trainingSet[i, j] = data[i, j];
                }
            }

            // Fill evaluation set
            for (int i = trainingRows; i < totalRows; i++)
            {
                for (int j = 0; j < totalColumns; j++)
                {
                    evaluationSet[i - trainingRows, j] = data[i, j];
                }
            }

            return (trainingSet, evaluationSet);
        }

        public static T DeepClone<T>(T obj)
        {
            return JsonConvert.DeserializeObject<T>(JsonConvert.SerializeObject(obj));
        }
    }
}
