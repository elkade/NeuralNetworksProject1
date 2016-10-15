using System.Linq;

namespace NeuralNetworksProject1
{
    class NormalizationHelper
    {
        //private readonly double _min;
        //private readonly double _max;
        private double _scale;
        private double _transition;

        public NormalizationHelper(double[][] data)
        {
            double max = data.Select(d => d.Max()).Max();
            double min = data.Select(d => d.Min()).Min();

            _scale = max > -min ? max : -min;//max - min;
            _transition = 0;//min;
        }
        public double Normalize(double value)
        {
            return (value - _transition) / _scale;
        }

        public double Denormalize(double value)
        {
            return value * _scale + _transition;
        }
    }
}
