using Accord.Statistics;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeuralNetPOC
{
    public class MichaelMaths
    {
        public static double GetStandardDeviation(double[] values)
        {
            double standardDeviation = 0;

            var mean = values.Average();


            for (int i = 0; i < values.Length; i++)
            {
                var value = values[i] - mean;
                standardDeviation += value * value;
            }

            standardDeviation /= values.Length;

            standardDeviation = Math.Sqrt(standardDeviation);

            return standardDeviation;
        }

        public static double GetVectorMagnitude(double[] values)
        {

            double result = 0;
            foreach (var value in values)
            {
                result += value * value;

            }

            result = Math.Sqrt(result);

            return Math.Abs(result);

        }

        public static double[] ScaleVector(double[] vector, double scalar)
        {
            for (int i = 0; i < vector.Length; i++)
            {
                vector[i] *= scalar;
            }

            return vector;
        }


        public static double GetDotProduct(double[] vectorA, double[] vectorB)
        {
            if (vectorA.Length != vectorB.Length)
            {
                throw new Exception("Vector size mismatch");
            }

            double result = 0;

            for (int i = 0; i < vectorA.Length; i++)
            {
                result += vectorA[i] * vectorB[i];
            }


            return result;
        }
        public static double GetDotProduct(double[] vectorA, double[] vectorB, double angleInDegrees)
        {
            if (vectorA.Length != vectorB.Length)
            {
                throw new Exception("Vector size mismatch");
            }

            double magnitudeA = GetVectorMagnitude(vectorA);
            double magnitudeB = GetVectorMagnitude(vectorB);

            double angleInRadians = DegreesToRadians(angleInDegrees);

            double dotProduct = magnitudeA * magnitudeB * Math.Cos(angleInRadians);

            return dotProduct;
        }

        public static double DegreesToRadians(double degrees)
        {
            return degrees * (Math.PI / 180);
        }

        public static double[] TestNormalisation(double[] vector)
        {
            var max = vector.Max();
            for (int i = 0; i < vector.Length; i++)
            {
                vector[i] = vector[i] / max;
            }
            return vector;
        }

        public static double[] NormalizeVector(double[] vector)
        {

            double magnitude = 0;
            for (int i = 0; i < vector.Length; i++)
            {
                magnitude += vector[i] * vector[i]; 
            }
            magnitude = Math.Sqrt(magnitude); 

            double[] normalizedVector = new double[vector.Length];
            for (int i = 0; i < vector.Length; i++)
            {
                normalizedVector[i] = vector[i] / magnitude; 
            }

            return normalizedVector;
        }

        public static double[] StandardizeVector(double[] vector)
        {
            double mean = vector.Average();
            double stdDev = Math.Sqrt(vector.Average(v => Math.Pow(v - mean, 2)));

            double[] standardizedVector = new double[vector.Length];
            for (int i = 0; i < vector.Length; i++)
            {
                standardizedVector[i] = (vector[i] - mean) / stdDev;
            }

            return standardizedVector;
        }

        public static double[] MinMaxNormalize(double[] vector)
        {
            double min = vector.Min();
            double max = vector.Max();
            return vector.Select(x => (x - min) / (max - min)).ToArray();
        }

        public static double MinMaxDenormalize(double value, double min, double max)
        {
            return (value * (max - min)) + min;
        }

        public static double[] ZScoreNormalize(double[] vector)
        {
            double mean = vector.Mean();
            double stdDev = GetStandardDeviation(vector);
            return vector.Select(x => (x - mean) / stdDev).ToArray();
        }
        public static double[] DenormalizeZScore(double[] normalizedVector, double mean, double stdDev)
        {
            return normalizedVector.Select(x => (x * stdDev) + mean).ToArray();
        }

        public static double[] MaxNormalize(double[] vector)
        {
            double maxAbs = vector.Max(Math.Abs);
            return vector.Select(x => x / maxAbs).ToArray();
        }

        public static double[] L2Normalize(double[] vector)
        {
            double l2Norm = Math.Sqrt(vector.Sum(x => x * x));
            return vector.Select(x => x / l2Norm).ToArray();
        }

        public static double[] L1Normalize(double[] vector)
        {
            double l1Norm = vector.Sum(Math.Abs);
            return vector.Select(x => x / l1Norm).ToArray();
        }

        public static double[] RobustScale(double[] vector)
        {
            double median = vector.OrderBy(x => x).Skip(vector.Length / 2).First();
            double q1 = vector.OrderBy(x => x).Skip(vector.Length / 4).First();
            double q3 = vector.OrderBy(x => x).Skip(3 * vector.Length / 4).First();
            double iqr = q3 - q1;


            // Handle the case where IQR is zero (e.g., all elements are identical)
            if (iqr == 0)
            {
                return vector.Select(x => 0.0).ToArray();
            }

            // Add a small epsilon to prevent division by zero
            const double epsilon = 1e-8;
            iqr = Math.Max(iqr, epsilon);

            return vector.Select(x => (x - median) / iqr).ToArray();
        }

        public static double[] MeanNormalize(double[] vector)
        {
            double mean = vector.Average();
            double min = vector.Min();
            double max = vector.Max();
            const double epsilon = 1e-8; // Small constant to prevent instability
            return vector.Select(x => (x - mean) / (max - min + epsilon)).ToArray();
        }

        public static double[] DecimalScale(double[] vector)
        {
            double maxAbs = vector.Max(Math.Abs);
            int j = (int)Math.Ceiling(Math.Log10(maxAbs));
            return vector.Select(x => x / Math.Pow(10, j)).ToArray();
        }

        public static double[,] ShuffleMatrix(double[,] matrix)
        {
            Random rand = new Random();
            int rowCount = matrix.GetLength(0);
            int colCount = matrix.GetLength(1);

            // Create a list of row indices to shuffle
            List<int> indices = Enumerable.Range(0, rowCount).ToList();
            indices = indices.OrderBy(x => rand.Next()).ToList(); // Shuffle the indices

            // Create a new matrix to store the shuffled rows
            double[,] shuffledMatrix = new double[rowCount, colCount];

            // Copy rows from the original matrix to the shuffled matrix
            for (int newRow = 0; newRow < rowCount; newRow++)
            {
                int originalRow = indices[newRow];
                for (int col = 0; col < colCount; col++)
                {
                    shuffledMatrix[newRow, col] = matrix[originalRow, col];
                }
            }

            return shuffledMatrix;
        }

        // Activation Functions
        public static double Sigmoid(double x)
        {
            return 1 / (1 + Math.Exp(-x));
        }

        public static double SigmoidDerivative(double x)
        {
            return Sigmoid(x) * (1 - Sigmoid(x));
        }

        public static double ReLU(double x)
        {
            return Math.Max(0, x); // If x > 0, return x; otherwise, return 0
        }

        public static double ReLUDerivative(double x)
        {
            return x > 0 ? 1 : 0; // Derivative is 1 for x > 0, and 0 otherwise
        }

        public static double LeakyReLU(double x, double alpha = 0.01)
        {
            return x > 0 ? x : alpha * x; // If x > 0, return x; otherwise, return alpha * x
        }

        public static double LeakyReLUDerivative(double x, double alpha = 0.01)
        {
            return x > 0 ? 1 : alpha; // Derivative is 1 for x > 0, and alpha otherwise
        }
    }
}
