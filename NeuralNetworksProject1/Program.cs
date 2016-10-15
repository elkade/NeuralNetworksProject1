﻿using Encog.ML.Data;
using Encog.ML.Data.Basic;
using Encog.Util.CSV;
using System.Collections.Generic;
using System.Linq;
using System;
using Encog.Neural.Networks.Layers;
using Encog.Engine.Network.Activation;
using Encog.Neural.Networks;
using Encog.Neural.Networks.Training.Propagation.Back;
using Encog.ML.Train;
using System.Text;
using System.Threading;
using System.IO;
using System.Globalization;

namespace NeuralNetworksProject1
{
    class Program
    {
        static double _scale;
        static double _transition;
        static void Main(string[] args)
        {
            //DoClassification();
            DoRegression();
        }

        private static void DoRegression()
        {
            int rowLength = 2;

            var trainingData = GetTrainingDataSet(@"DataSets/data.xsq.train.csv", rowLength);

            var network = CreateNetwork(rowLength - 1);

            IMLTrain training = new Backpropagation(network, trainingData);

            Train(training);

            var testData = GetTestDataSet(@"DataSets/data.xsq.test.csv", rowLength - 1);

            //Test(trainingData, network);

            Visualize(testData, network, @"DataSets/data.xsq.output.csv", @"DataSets/data.xsq.train.csv");
        }

        private static void DoClassification()
        {
            int rowLength = 3;

            var trainingData = GetTrainingDataSet(@"DataSets/data.train.csv", rowLength);

            var network = CreateNetwork(rowLength - 1);

            IMLTrain training = new Backpropagation(network, trainingData);

            Train(training);

            var testData = GetTestDataSet(@"DataSets/data.test.csv", rowLength - 1);

            Test(trainingData, network);
        }

        private static void Visualize(IMLDataSet normalizedData, BasicNetwork network, string outputPath, string inputPath)
        {
            var csv = new StringBuilder();
            foreach (var row in normalizedData)
            {
                var sb = new StringBuilder();

                IMLData output = network.Compute(row.Input);

                var result = Denormalize(output[0]);

                StringBuilder rowBuilder = new StringBuilder();

                for (int i = 0; i < row.Input.Count; i++)
                {
                    rowBuilder.Append(Denormalize(row.Input[i]).ToString(CultureInfo.GetCultureInfo("en-GB")));
                    rowBuilder.Append(",");
                }
                rowBuilder.Append(result.ToString(CultureInfo.GetCultureInfo("en-GB")));
                csv.AppendLine(rowBuilder.ToString());

            }
            File.WriteAllText(outputPath, csv.ToString());
            RScriptRunner.RunFromCmd("PlotScript.R", "RScript.exe", outputPath, inputPath);
        }

        private static IMLDataSet GetTestDataSet(string path, int rowLength)
        {
            double[][] data;
            var format = new CSVFormat('.', ',');
            var csv = new ReadCSV(path, true, format);
            data = GetData(csv, rowLength).ToArray();
            csv.Close();

            data = Normalize(data);
            return new BasicMLDataSet(data, data.Select(d=>new double[0]).ToArray()) ;
        }

        private static void Test(IMLDataSet normalizedData, BasicNetwork network)
        {
            foreach (var row in normalizedData)
            {
                var sb = new StringBuilder();

                IMLData output = network.Compute(row.Input);

                var result = Denormalize(output[0]);

                sb.Append(" -> predicted: ");
                sb.Append(result);

                if (row.Ideal.Count > 0)
                {
                    var correct = Denormalize(row.Ideal[0]);
                    sb.Append("(correct: ");
                    sb.Append(correct);
                    sb.Append(")");
                }
                Console.WriteLine(sb.ToString());
            }
        }

        private static void Train(IMLTrain training)
        {
            int epoch = 1;
            do
            {
                training.Iteration();
                Console.WriteLine(@"Epoch #" + epoch + @" Error:" + training.Error);
                epoch++;
            } while (training.Error > 0.002);

            training.FinishTraining();
        }

        private static BasicNetwork CreateNetwork(int inputSize)
        {
            var network = new BasicNetwork();
            network.AddLayer(new BasicLayer(null, true, inputSize));
            network.AddLayer(new BasicLayer(new ActivationSigmoid(), true, 10));
            network.AddLayer(new BasicLayer(new ActivationElliott(), true, 20));
            network.AddLayer(new BasicLayer(new ActivationSigmoid(), true, 1));
            network.Structure.FinalizeStructure();
            network.Reset();
            return network;
        }

        static IEnumerable<double[]> GetData(ReadCSV csv, int length)
        {
            while (csv.Next())
            {
                double[] row = new double[length];

                for (int i = 0; i < length; i++)
                    row[i] = csv.GetDouble(i);

                yield return row;
            }
        }

        private static IMLDataSet GetTrainingDataSet(string path, int rowLength)
        {
            double[][] data;
            var format = new CSVFormat('.', ',');
            var csv = new ReadCSV(path, true, format);
            data = GetData(csv, rowLength).ToArray();
            csv.Close();

            data = Normalize(data);

            double[][] input = data.Select(d => d.Take(rowLength - 1).ToArray()).ToArray();
            double[][] output = data.Select(d => d.Skip(rowLength - 1).ToArray()).ToArray();

            return new BasicMLDataSet(input, output);
        }

        private static double[][] Normalize(double[][] data)
        {
            double max = data.Select(d => d.Max()).Max();
            double min = data.Select(d => d.Min()).Min();

            _scale = max - min;
            _transition = min;
            return data.Select(d1 => d1.Select(d2 => (d2 - _transition) / _scale).ToArray()).ToArray();
        }

        private static double Denormalize(double value)
        {
            return value * _scale + _transition;
        }
    }
}
