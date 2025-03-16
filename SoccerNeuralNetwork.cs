using System;
using System.Collections.Generic;
using Accord.Math;
using System.Diagnostics;
namespace NeuralNetPOC
{
    
    public class SoccerNeuralNetwork
    {

        public ActivationFunctionType activationFunctionType;
        private LearningRateDecayType learningRateDecayType = LearningRateDecayType.None;
        // Weight and Bias
        static Random rand = new Random();
        List<double[,]> hiddenWeights;
        double[,] outputWeights;
        private double[] outputBiases;
        List<double[]> hiddenBiases;
        private int[] hiddenLayerSize;
        bool allowEarlyExit = true;
        int patience = 1000; // Number of epochs to wait for improvement
        double bestError = double.MaxValue;
        int epochsWithoutImprovement = 0;
        double decayRate = 0.01;

        public int outputSize;
        public List<double[,]> bestHiddenWeights;
        public double[,] bestOutputWeights;
        public List<double[]> bestHiddenBiases;
        public double[] bestOutputBiases;


        List<double[]> hiddenOutputs;

        public double[,] _embeddingMatrix;
        private Dictionary<double, string> matrixLookup = new Dictionary<double, string>();
        private int _embeddingDepth;
        public Dictionary<string, double[]> embeddingMatrixDict = new Dictionary<string, double[]>();


        // creat embedding matrix to handle categorical data / text data, that needs to be converted to numbers
        public double[,] CreateEmbeddingMatrix(string[] dataToEmbed, int embeddingDepth)
        {
            _embeddingDepth = embeddingDepth; // Adjust based on the rule of thumb
            double[,] embeddingMatrix = new double[dataToEmbed.Length, embeddingDepth];

            // Randomly initialize embeddings
            Random random = new Random();
            for (int i = 0; i < dataToEmbed.Length; i++)
            {
                for (int j = 0; j < _embeddingDepth; j++)
                {
                    embeddingMatrix[i, j] = random.NextDouble(); // Random values between 0 and 1
                }

                matrixLookup.Add(embeddingMatrix.GetRow(i).Sum(), dataToEmbed[i]);
                embeddingMatrixDict.Add(dataToEmbed[i], embeddingMatrix.GetRow(i));
            }
            _embeddingMatrix = embeddingMatrix;
            return embeddingMatrix;
        }

        public double[,] GetEmbeddingMatrix()
        {
            return _embeddingMatrix;
        }

        public enum ActivationFunctionType
        {
            Sigmoid,
            ReLU,
            LeakyReLU
        }

        public enum LearningRateDecayType
        {
            None,
            TimeBased,
            Exponential,
            Step,
            Polynominal,
            Cosine,
        }
       

        public void Evaluate(double[,] evaluationData)
        {
            double mae = 0;

            for (int i = 0; i < evaluationData.Rows(); i++)
            {
                // Extract the inputs (last 7 columns are inputs)
                double[] inputs = evaluationData.GetRow(i).TakeLast(7).ToArray();

                double actualTeam1Goals = evaluationData.GetRow(i)[0];
                double actualTeam2Goals = evaluationData.GetRow(i)[1];
                double actualTeam1Corners = evaluationData.GetRow(i)[2];
                double actualTeam2Corners = evaluationData.GetRow(i)[3];

                // Predict using the model
                double[] predictedValues = Predict(inputs);

                // Find corresponding action from original dataset
                string action = matrixLookup[evaluationData.GetRow(i).TakeLast(_embeddingDepth).ToArray().Sum()];

                // Log the results
                Debug.WriteLine($"Action: {action}");
                Debug.WriteLine($"Actual Team 1 Goals: {actualTeam1Goals} -----  Predicted: {predictedValues[0]}");
                Debug.WriteLine($"Actual Team 2 Goals: {actualTeam2Goals} -----  Predicted: {predictedValues[1]}");
                Debug.WriteLine($"Actual Team 1 Corners: {actualTeam1Corners} -----  Predicted: {predictedValues[2]}");
                Debug.WriteLine($"Actual Team 2 Corners: {actualTeam2Corners} -----  Predicted: {predictedValues[3]}");
                Debug.WriteLine($"Actual Previous Team 1 Goals: {evaluationData.GetRow(i)[4]}");
                Debug.WriteLine($"Actual Previous Team 2 Goals: {evaluationData.GetRow(i)[5]}");
                Debug.WriteLine($"Actual Previous Team 1 Corners: {evaluationData.GetRow(i)[6]}");
                Debug.WriteLine($"Actual Previous Team 2 Corners: {evaluationData.GetRow(i)[7]}");
                if (actualTeam1Goals != (int)Math.Round(predictedValues[0])
                    || actualTeam2Goals != (int)Math.Round(predictedValues[1])
                    || actualTeam1Corners != (int)Math.Round(predictedValues[2])
                    || actualTeam2Corners != (int)Math.Round(predictedValues[3]))
                {
                    Debug.WriteLine("INCORRECT PREDICTION IN THIS ROW");
                }
                Debug.WriteLine("");

                // Calculate MAE (Mean Absolute Error)
                // the maths here is bad, could end up with the equation occilating between positive and negative error values which when summed up are very inaccurate
                // do math abs on each subtraction independently at some point and check it works better
                mae += Math.Abs((predictedValues[0] - actualTeam1Goals)
                              + (predictedValues[1] - actualTeam2Goals)
                              + (predictedValues[2] - actualTeam1Corners)
                              + (predictedValues[3] - actualTeam2Corners));
            }

            // Compute final MAE
            mae /= evaluationData.Rows();
            Debug.WriteLine($"Mean Absolute Error on Evaluation Set: {mae}");
        }

        public void Train(double[,] inputs, double[,] outputs, int[] _hiddenLayerSize, double initialLearningRate, int epochs, ActivationFunctionType activationType)
        {
            int nSamples = inputs.GetLength(0); // Number of training samples (rows)
            int nFeatures = inputs.GetLength(1); // Number of features (columns)

            // Update activation function type globally for the network
            activationFunctionType = activationType;

            // Initialize the network with the specified hidden layers
            InitializeNetwork(inputs, outputs, _hiddenLayerSize, activationType);
            double learningRate = initialLearningRate;
            for (int epoch = 0; epoch < epochs; epoch++)
            {
                double totalError = 0;

                //uncomment below if you wanna apply learning rate altercation constantly but will probs not go well
                //double learningRate = ApplyLearningRate(initialLearningRate, epoch, epochs);

                for (int sample = 0; sample < nSamples; sample++)
                {
                    // Forward pass
                    ComputeHiddenOutputs(inputs, sample);
                    double[] predictedOutputs = ComputeOutputLayer(hiddenOutputs.Last());

                    // Drop neurons every now and again to rebalance things, although i think this is not implemented ideally
                    hiddenOutputs[hiddenOutputs.Count - 1] = DropNeurons(hiddenOutputs.Last());

                    double[] errors = ComputeErrors(predictedOutputs, outputs, sample);
                    totalError += CalculateTotalError(errors);

                    // Update layers
                    UpdateOutputLayer(hiddenOutputs.Last(), errors, learningRate);
                    UpdateHiddenLayers(inputs, sample, errors, learningRate);
                }

                Debug.WriteLine($"Epoch {epoch + 1}/{epochs}, Average Error: {totalError / nSamples:F17} epochs without improvement: {epochsWithoutImprovement}");
                if (totalError / nSamples < bestError)
                {
                    bestError = totalError / nSamples;

                    // store our best weights and bias to revert back to 
                    UpdateBestValues();
                }
                else
                {
                    epochsWithoutImprovement++;
                    //apply learning rate altercation if we arnt improving
                    learningRate = ApplyLearningRate(initialLearningRate, epoch, epochs);
                }

                // stop the training early if we are not making any progress
                if (epochsWithoutImprovement > patience)
                {
                    Debug.WriteLine($"Early stopping triggered. Will use best score of {bestError}");
                    return;
                }
            }
        }

        private void UpdateBestValues()
        {
            epochsWithoutImprovement = 0;
            bestHiddenWeights = Utils.DeepClone(hiddenWeights);
            bestOutputWeights = (double[,])outputWeights.Clone();
            bestHiddenBiases = Utils.DeepClone(hiddenBiases);
            bestOutputBiases = (double[])outputBiases.Clone();
        }

        private double ApplyLearningRate(double initialLearningRate, int epoch, int maxEpochs, double factor = 0.5, int stepSize = 10, int power = 2)
        {
            switch (learningRateDecayType)
            {
                case LearningRateDecayType.None:
                    return initialLearningRate;

                case LearningRateDecayType.TimeBased:
                    return initialLearningRate / (1 + decayRate * epoch);

                case LearningRateDecayType.Exponential:
                    return initialLearningRate * Math.Exp(-decayRate * epoch);

                case LearningRateDecayType.Step:
                    return initialLearningRate * Math.Pow(factor, epoch / stepSize);

                case LearningRateDecayType.Polynominal:
                    return initialLearningRate * Math.Pow(1 - (double)epoch / maxEpochs, power);

                case LearningRateDecayType.Cosine:
                    return initialLearningRate * 0.5 * (1 + Math.Cos((double)epoch / maxEpochs * Math.PI));

                default:
                    throw new ArgumentException("Unsupported decay type. Choose from: exponential, step, polynomial, cosine.");
            }
        }

        private void InitializeNetwork(double[,] inputs, double[,] outputs, int[] _hiddenLayerSize, ActivationFunctionType activationType)
        {
            activationFunctionType = activationType;
            hiddenLayerSize = _hiddenLayerSize;
            outputSize = outputs.GetLength(1);
            int nFeatures = inputs.GetLength(1);

            // Initialize hidden layer weights and biases
            hiddenWeights = new List<double[,]>(); // List to hold weights for each hidden layer
            hiddenBiases = new List<double[]>();   // List to hold biases for each hidden layer

            hiddenOutputs = new List<double[]>();
            foreach (int size in hiddenLayerSize)
            {
                hiddenOutputs.Add(new double[size]);
            }

            int previousLayerSize = nFeatures;
            foreach (int layerSize in _hiddenLayerSize)
            {
                // Initialize weights for the current hidden layer
                double[,] layerWeights = new double[layerSize, previousLayerSize];
                double[] layerBiases = new double[layerSize];

                for (int i = 0; i < layerSize; i++)
                {
                    for (int j = 0; j < previousLayerSize; j++)
                    {
                        layerWeights[i, j] = (rand.NextDouble() - 0.5) * Math.Sqrt(2.0 / previousLayerSize);
                    }
                    layerBiases[i] = (rand.NextDouble() - 0.5) * 0.1;
                }

                hiddenWeights.Add(layerWeights);
                hiddenBiases.Add(layerBiases);

                previousLayerSize = layerSize; // Update for next layer
            }

            // Initialize output layer weights and biases
            outputWeights = new double[outputSize, previousLayerSize];
            outputBiases = new double[outputSize];

            for (int i = 0; i < outputSize; i++)
            {
                for (int j = 0; j < previousLayerSize; j++)
                {
                    outputWeights[i, j] = (rand.NextDouble() - 0.5) * Math.Sqrt(2.0 / previousLayerSize);
                }
                outputBiases[i] = (rand.NextDouble() - 0.5) * 0.1;
            }
        }

        private double[] ComputeHiddenLayer(double[,] inputs, int sampleIndex)
        {
            // Extract the input features for the current sample
            int inputSize = inputs.GetLength(1);
            double[] currentInputs = new double[inputSize];

            for (int j = 0; j < inputSize; j++)
            {
                currentInputs[j] = inputs[sampleIndex, j];
            }

            // Forward pass through each hidden layer
            foreach (var (weights, biases) in hiddenWeights.Zip(hiddenBiases, Tuple.Create))
            {
                int numNeurons = biases.Length;
                double[] layerOutputs = new double[numNeurons];

                for (int i = 0; i < numNeurons; i++)
                {
                    double weightedSum = biases[i]; // Start with the bias
                    for (int j = 0; j < currentInputs.Length; j++)
                    {
                        weightedSum += weights[i, j] * currentInputs[j];
                    }
                    layerOutputs[i] = ApplyActivationFunction(activationFunctionType, weightedSum);
                }

                // Outputs of the current layer become inputs to the next
                currentInputs = layerOutputs;
            }

            return currentInputs; // Final outputs after all hidden layers
        }

        private double[] ComputeOutputLayer(double[] lastHiddenLayerOutputs)
        {
            double[] predictedOutputs = new double[outputSize];

            for (int i = 0; i < outputSize; i++)
            {
                double weightedSum = outputBiases[i]; 
                for (int j = 0; j < lastHiddenLayerOutputs.Length; j++)
                {
                    weightedSum += outputWeights[i, j] * lastHiddenLayerOutputs[j];
                }
                predictedOutputs[i] = weightedSum; // No activation function applied here for regression
            }

            return predictedOutputs;
        }

        private double[] ComputeErrors(double[] predictedOutputs, double[,] actualOutputs, int sampleIndex)
        {
            double[] errors = new double[outputSize];
            for (int i = 0; i < outputSize; i++)
            {
                errors[i] = predictedOutputs[i] - actualOutputs[sampleIndex, i];
            }
            return errors;
        }

        private double CalculateTotalError(double[] errors)
        {
            double totalError = 0.0;
            foreach (double error in errors)
            {
                totalError += Math.Pow(error, 2); // Sum of squared errors
            }
            return totalError / errors.Length; // Average squared error
        }


        // at some point sort this out to use SIMD to speed up processing
        private void UpdateHiddenLayers(double[,] inputs, int sampleIndex, double[] outputErrors, double learningRate)
        {
            // List to store gradients for each hidden layer, starting from the output layer errors
            List<double[]> layerGradients = new List<double[]>();
            double[] nextLayerErrors = outputErrors;

            // Backpropagate errors through each hidden layer, starting from the last hidden layer
            for (int layer = hiddenWeights.Count - 1; layer >= 0; layer--)
            {
                int currentLayerSize = hiddenWeights[layer].GetLength(0);
                int previousLayerSize = (layer == 0) ? inputs.GetLength(1) : hiddenWeights[layer - 1].GetLength(0);

                double[] currentLayerGradients = new double[currentLayerSize];

                for (int neuron = 0; neuron < currentLayerSize; neuron++)
                {
                    double errorSum = 0.0;

                    // Compute gradient by backpropagating errors from the next layer
                    for (int k = 0; k < nextLayerErrors.Length; k++)
                    {
                        errorSum += nextLayerErrors[k] * (layer == hiddenWeights.Count - 1 ? outputWeights[k, neuron] : hiddenWeights[layer + 1][k, neuron]);
                    }

                    errorSum *= ApplyActivationFunctionDerivative(activationFunctionType, hiddenOutputs[layer][neuron]);
                    currentLayerGradients[neuron] = errorSum;

                    // Update weights and biases for the current hidden layer
                    for (int j = 0; j < previousLayerSize; j++)
                    {
                        double input = (layer == 0) ? inputs[sampleIndex, j] : hiddenOutputs[layer - 1][j];
                        double weightGradient = errorSum * input;
                        hiddenWeights[layer][neuron, j] -= learningRate * ClipGradient(weightGradient, -5, 5);
                    }

                    hiddenBiases[layer][neuron] -= learningRate * errorSum;
                }

                layerGradients.Insert(0, currentLayerGradients);
                nextLayerErrors = currentLayerGradients; // Propagate errors to the previous layer
            }
        }

        private void UpdateOutputLayer(double[] lastHiddenLayerOutputs, double[] errors, double learningRate)
        {
            for (int i = 0; i < outputSize; i++)
            {
                double outputGradient = errors[i]; // No activation derivative needed for regression output
                for (int j = 0; j < lastHiddenLayerOutputs.Length; j++)
                {
                    // Calculate weight gradient and update
                    double weightGradient = outputGradient * lastHiddenLayerOutputs[j];
                    outputWeights[i, j] -= learningRate * weightGradient;
                }
                // Update bias
                outputBiases[i] -= learningRate * outputGradient;
            }
        }

        private void ComputeHiddenOutputs(double[,] inputs, int sampleIndex)
        {
            double[] currentInputs = new double[inputs.GetLength(1)];
            for (int j = 0; j < inputs.GetLength(1); j++)
            {
                currentInputs[j] = inputs[sampleIndex, j];
            }

            for (int layer = 0; layer < hiddenLayerSize.Length; layer++)
            {
                double[] layerOutputs = new double[hiddenLayerSize[layer]];

                for (int neuron = 0; neuron < hiddenLayerSize[layer]; neuron++)
                {
                    double weightedSum = hiddenBiases[layer][neuron];
                    for (int inputIndex = 0; inputIndex < currentInputs.Length; inputIndex++)
                    {
                        weightedSum += hiddenWeights[layer][neuron, inputIndex] * currentInputs[inputIndex];
                    }
                    layerOutputs[neuron] = ApplyActivationFunction(activationFunctionType, weightedSum);
                }

                hiddenOutputs[layer] = layerOutputs; // Store outputs for this layer
                currentInputs = layerOutputs; // Input to the next layer is the output of this one
            }
        }

        private double[] DropNeurons(double[] hiddenOutputs)
        {
            for (int i = 0; i < hiddenOutputs.Length; i++)
            {
                if (rand.NextDouble() < 0.2)
                {
                    hiddenOutputs[i] = 0; // Drop this neuron
                }
            }
            return hiddenOutputs;
        }

        private double ClipGradient(double gradient, double min, double max)
        {
            return Math.Max(min, Math.Min(max, gradient));
        }

        public double[] Predict(double[] inputs)
        {
            double[] currentInputs = inputs;

            foreach (var (weights, biases) in bestHiddenWeights.Zip(bestHiddenBiases, Tuple.Create))
            {
                currentInputs = ComputeLayerOutput(currentInputs, weights, biases);
            }

            double[] predictedOutputs = new double[outputSize];
            for (int i = 0; i < outputSize; i++)
            {
                double weightedSum = bestOutputBiases[i];
                for (int j = 0; j < currentInputs.Length; j++)
                {
                    weightedSum += bestOutputWeights[i, j] * currentInputs[j];
                }
                predictedOutputs[i] = weightedSum; 
            }

            return predictedOutputs;
        }

        private double[] ComputeLayerOutput(double[] inputs, double[,] weights, double[] biases)
        {
            int numNeurons = weights.GetLength(0);
            double[] outputs = new double[numNeurons];

            for (int i = 0; i < numNeurons; i++)
            {
                double weightedSum = biases[i];
                for (int j = 0; j < inputs.Length; j++)
                {
                    weightedSum += weights[i, j] * inputs[j];
                }
                outputs[i] = ApplyActivationFunction(activationFunctionType, weightedSum);
            }

            return outputs;
        }

        public double ApplyActivationFunction(ActivationFunctionType type, double value)
        {
            switch (type)
            {
                case ActivationFunctionType.Sigmoid:
                    return MichaelMaths.Sigmoid(value);
                case ActivationFunctionType.ReLU:
                    return MichaelMaths.ReLU(value);
                case ActivationFunctionType.LeakyReLU:
                    return MichaelMaths.LeakyReLU(value);
                default:
                    throw new NotImplementedException();
            }
        }

        public double ApplyActivationFunctionDerivative(ActivationFunctionType type, double value)
        {
            switch (type)
            {
                case ActivationFunctionType.Sigmoid:
                    return MichaelMaths.SigmoidDerivative(value);
                case ActivationFunctionType.ReLU:
                    return MichaelMaths.ReLUDerivative(value);
                case ActivationFunctionType.LeakyReLU:
                    return MichaelMaths.LeakyReLUDerivative(value);
                default:
                    throw new NotImplementedException();
            }
        }
    }
}
