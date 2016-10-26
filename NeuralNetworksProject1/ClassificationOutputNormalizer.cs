using Encog.ML.Data;
using System;
using System.Linq;

namespace NeuralNetworksProject1
{
    class ClassificationOutputNormalizer
    {
        int _outputSize;
        public ClassificationOutputNormalizer(int outputSize)
        {
            _outputSize = outputSize;
        }
        public double[] Normalize(double value)
        {
            var output = new double[_outputSize];
            output[(int)Math.Round(value - 1)] = 1f;
            return output;
        }

        public double Denormalize(IMLData data)
        {
            double[] values = new double[_outputSize];

            for (int i = 0; i < _outputSize; i++)
                values[i] = data[i];

            if (values.Length != _outputSize)
                throw new ArgumentException();
            double maxValue = values.Max();
            int maxIndex = values.ToList().IndexOf(maxValue);
            return maxIndex + 1;
        }
    }
}
