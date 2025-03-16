using Accord.Math;
using Newtonsoft.Json;
using NUnit.Framework;
using System.Diagnostics;
using static NeuralNetPOC.SoccerNeuralNetwork;

namespace NeuralNetPOC
{

    internal class SoccerTest
    {

        [Test]
        public void TrainModel()
        {
            // generate x amount of random training data
            var dataSet = new SoccerModelTrainer().GenerateTrainingData(10000);


            // Define actions and create action-to-index mapping
            var actions = dataSet.Select(x => x.Action).Distinct().ToList(); // Get all unique actions
            var actionToIndex = actions.Select((action, index) => new { action, index })
                                       .ToDictionary(x => x.action, x => x.index);

            var nn = new SoccerNeuralNetwork();
            nn.CreateEmbeddingMatrix(actions.ToArray(), 3); // Create embeddings

            // Prepare dataset with embeddings
            double[,] data = new double[dataSet.Count, dataSet.First().GetAttributesArray().Count() + nn.GetEmbeddingMatrix().GetRow(0).Length];

            //load all our prepared data into the table
            for (int i = 0; i < data.GetLength(0); i++)
            {
                var rowData = dataSet[i].GetAttributesArray(); // Extract numerical attributes
                int actionIndex = actionToIndex[dataSet[i].Action]; // Find the correct row in the embedding matrix
                var embedding = nn.GetEmbeddingMatrix().GetRow(actionIndex); // Get the embedding row

                data.SetRow(i, rowData.Concat(embedding).ToArray()); // Combine numerical features with embedding
            }


            // Split dataset into training and evaluation sets ratio
            (double[,] trainingSet, double[,] evaluationSet) = Utils.SplitDataset(data, 0.9);

            // Extract inputs and outputs from the training set
            double[,] inputs = trainingSet.GetColumns(4, 5, 6, 7, 8, 9, 10); // Previous goals/corners + action embedding
            double[,] outputs = trainingSet.GetColumns(0, 1, 2, 3);         // Team 1 Goals, Team 2 Goals, Team 1 Corners, Team 2 Corners

            // Train the neural network
            nn.Train(inputs, outputs, new int[] { 18 }, 0.00001, 100000, ActivationFunctionType.LeakyReLU);
            // Evaluate the neural network
            nn.Evaluate(evaluationSet);
            
            // save the trained model for future use
            var serialisedData = JsonConvert.SerializeObject(nn);
            var path = Environment.CurrentDirectory + @"\models\currentModel.json"; // probs shouldnt have this in the bin directory but meh
            File.WriteAllText(path, serialisedData);

        }

        [Test]
        public void SoccerRun()
        {
            // fetch our pre-trained model
            var neuralNetwork = JsonConvert.DeserializeObject<SoccerNeuralNetwork>(File.ReadAllText(Environment.CurrentDirectory + @"..\..\savedmodels\currentModelBest.json"));
            //var neuralNetwork = JsonConvert.DeserializeObject<SoccerNeuralNetwork>(File.ReadAllText(Environment.CurrentDirectory + @"\models\currentModel.json"));

            // new up a model
            var soccerData = new SoccerModel(false);

            bool problemFound = false;
            Random random = new Random();
            while (problemFound == false)
            {
                int randomNumber = random.Next(0, 6);

                switch (randomNumber)
                {
                    case 0:
                        soccerData.AddTeam1Goal();
                        break;
                    case 1:
                        soccerData.AddTeam2Goal();
                        break;
                    case 2:
                        soccerData.AddTeam1Corner();
                        break;
                    case 3:
                        soccerData.AddTeam2Corner();
                        break;
                    case 4:
                        soccerData.AddTeam1GoalFromCorner();
                        break;
                    case 5:
                        soccerData.AddTeam2GoalFromCorner();
                        break;
                    default:
                        throw new NotImplementedException();
                }
                
                // convert data from the soccer model into the format the predictor needs
                var data = new List<double>();
                data.AddRange(soccerData.GetAttributesArray().TakeLast(4).ToList());
                data.AddRange(neuralNetwork.embeddingMatrixDict[soccerData.Action]);

                // predict and round the values
                //var predictedValues = neuralNetwork.Predict(data.ToArray());
                //predictedValues = predictedValues.Select(x=> Math.Round(x)).ToArray();

                //var predictedValues = neuralNetwork.Predict(data.ToArray()).Select(x => Math.Abs(Math.Round(x))).ToArray();
                var predictedValues = neuralNetwork.Predict(data.ToArray());
                var predictedValuesRounded = predictedValues.Select(x => Math.Abs(Math.Round(x))).ToArray();
                //predictedValues = predictedValues.Select(x => Math.Round(x)).ToArray();

                // Log the results
                Debug.WriteLine("-----");
                Debug.WriteLine($"Action: {soccerData.Action}");
                Debug.WriteLine($"Actual Team 1 Goals: {soccerData.Team1Goals} -----  Predicted: {predictedValuesRounded[0]}");
                Debug.WriteLine($"Actual Team 2 Goals: {soccerData.Team2Goals} -----  Predicted: {predictedValuesRounded[1]}");
                Debug.WriteLine($"Actual Team 1 Corners: {soccerData.Team1Corners} -----  Predicted: {predictedValuesRounded[2]}");
                Debug.WriteLine($"Actual Team 2 Corners: {soccerData.Team2Corners} -----  Predicted: {predictedValuesRounded[3]}");
                Debug.WriteLine($"Actual Previous Team 1 Goals: {soccerData.PreviousTeam1Goals}");
                Debug.WriteLine($"Actual Previous Team 2 Goals: {soccerData.PreviousTeam2Goals}");
                Debug.WriteLine($"Actual Previous Team 1 Corners: {soccerData.PreviousTeam1Corners}");
                Debug.WriteLine($"Actual Previous Team 2 Corners: {soccerData.PreviousTeam2Corners}");
                Debug.WriteLine("-----");

                // if any of the actual new values dont match the predicted values then flag up a problem
                if (!predictedValuesRounded.SequenceEqual(soccerData.GetAttributesArray().Take(4)))
                {
                    problemFound = true;
                    Debug.WriteLine("PROBLEM FOUND");
                    Assert.Fail("Actual values did not match predicted values, potential bug found");
                }
                Thread.Sleep(3000);
            }



        }
    }
}
