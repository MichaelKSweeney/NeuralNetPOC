using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeuralNetPOC
{
    public class SoccerModel : ICloneable
    {
        public SoccerModel(bool generateRandomInitialValues)
        {
            Random random = new Random();
            if (generateRandomInitialValues)
            {
                PreviousTeam1Goals = random.Next(0, 6);
                PreviousTeam2Goals = random.Next(0, 6);
                PreviousTeam1Corners = random.Next(0, 6);
                PreviousTeam2Corners = random.Next(0, 6);
                Team1Goals = PreviousTeam1Goals;
                Team2Goals = PreviousTeam2Goals;
                Team1Corners = PreviousTeam1Corners;
                Team2Corners = PreviousTeam2Corners;
            }
        }

        public void AddTeam1Goal()
        {
            setPreviousValues();
            Team1Goals++;
            Action = "Team 1 Goal";
        }

        public void AddTeam2Goal()
        {
            setPreviousValues();
            Team2Goals++;
            Action = "Team 2 Goal";
        }

        public void AddTeam1Corner()
        {
            setPreviousValues();
            Team1Corners++;
            Action = "Team 1 Corner";
        }

        public void AddTeam2Corner()
        {
            setPreviousValues();
            Team2Corners++;
            Action = "Team 2 Corner";
        }

        public void AddTeam1GoalFromCorner()
        {
            setPreviousValues();
            Team1Goals++;
            Team1Corners++;
            Action = "Team 1 Goal From Corner";
        }

        public void AddTeam2GoalFromCorner()
        {
            setPreviousValues();
            Team2Goals++;
            Team2Corners++;
            Action = "Team 2 Goal From Corner";
        }

        private void setPreviousValues()
        {
            PreviousTeam1Goals = Team1Goals;
            PreviousTeam2Goals = Team2Goals;
            PreviousTeam1Corners = Team1Corners;
            PreviousTeam2Corners = Team2Corners;
        }

        public double[] GetAttributesArray()
        {
            return new double[]
            {
                Team1Goals,
                Team2Goals,
                Team1Corners,
                Team2Corners,
                PreviousTeam1Goals,
                PreviousTeam2Goals,
                PreviousTeam1Corners,
                PreviousTeam2Corners
            };
        }
        //new values
        public string Action { get; set; }
        public double Team1Goals { get; set; }
        public double Team2Goals { get; set; }
        public double Team1Corners { get; set; }
        public double Team2Corners { get; set; }

        //old values
        public double PreviousTeam1Goals { get; set; }
        public double PreviousTeam2Goals { get; set; }
        public double PreviousTeam1Corners { get; set; }
        public double PreviousTeam2Corners { get; set; }

        public object Clone()
        {
            return this.MemberwiseClone();
        }
    }

    public class SoccerModelTrainer
    {

        public void ConvertToMatrix(List<SoccerModel> data)
        {
            var matrix = new double[data.Count, 4];

            for (int i = 0; i < data.Count; i++)
            {

                matrix[i, 0] = data[i].Team1Goals;
                matrix[i, 1] = data[i].Team2Goals;
                matrix[i, 2] = data[i].PreviousTeam1Goals;
                matrix[i, 3] = data[i].PreviousTeam2Goals;
            }
        }

        //public List<SoccerInputModel> GenerateSequentialTrainingData(int entries)
        //{
        //    List<SoccerInputModel> events = new List<SoccerInputModel>
        //    {
        //        new SoccerInputModel(false) { Action = "Kick Off" }
        //    };

        //    Random random = new Random();
        //    for (int i = 0; i < entries - 1; i++)
        //    {
        //        int randomNumber = random.Next(0, 2);

        //        //var newEvent = new SoccerModel();
        //        var newEvent = (SoccerInputModel)events.Last().Clone();

        //        newEvent.PreviousTeam1Goals = events.Last().Team1Goals;
        //        newEvent.PreviousTeam2Goals = events.Last().Team2Goals;

        //        switch (randomNumber)
        //        {
        //            case 0:
        //                newEvent.Team1Goals++;
        //                newEvent.Action = "Team 1 Goal";
        //                break;
        //            case 1:
        //                newEvent.Team2Goals++;
        //                newEvent.Action = "Team 2 Goal";
        //                break;
        //            //case 2:
        //            //    newEvent.Team1Corners++;
        //            //    newEvent.Action = "Team 1 Corner";
        //            //    break;
        //            //case 3:
        //            //    newEvent.Team2Corners++;
        //            //    newEvent.Action = "Team 2 Corner";
        //            //    break;
        //            //case 4:
        //            //    newEvent.Team1Goals++;
        //            //    newEvent.Team1Corners++;
        //            //    newEvent.Action = "Team 1 Goal From Corner";
        //            //    break;
        //            //case 5:
        //            //    newEvent.Team2Goals++;
        //            //    newEvent.Team2Corners++;
        //            //    newEvent.Action = "Team 2 Goal From Corner";
        //            //    break;
        //            default:
        //                throw new NotImplementedException();
        //        }
        //        events.Add(newEvent);
        //    }
        //    return events;
        //}

        public List<SoccerModel> GenerateTrainingData(int entries)
        {
            List<SoccerModel> events = new List<SoccerModel>();

            Random random = new Random();
            for (int i = 0; i < entries - 1; i++)
            {
                int randomNumber = random.Next(0, 6);

                var newEvent = new SoccerModel(true);

                switch (randomNumber)
                {
                    case 0:
                        newEvent.AddTeam1Goal();
                        break;
                    case 1:
                        newEvent.AddTeam2Goal();
                        break;
                    case 2:
                        newEvent.AddTeam1Corner();
                        break;
                    case 3:
                        newEvent.AddTeam2Corner();
                        break;
                    case 4:
                        newEvent.AddTeam1GoalFromCorner();
                        break;
                    case 5:
                        newEvent.AddTeam2GoalFromCorner();
                        break;
                    default:
                        throw new NotImplementedException();
                }
                events.Add(newEvent);
            }
            return events;
        }
    }
}
