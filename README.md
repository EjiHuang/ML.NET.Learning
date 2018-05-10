# ML.NET.Learning
Learning ML.NET because i like .net.

test:

using Microsoft.ML;
using Microsoft.ML.Runtime.Api;
using Microsoft.ML.Trainers;
using Microsoft.ML.Transforms;
using System;

namespace ML.NET.CoreDemo
{
    class Program
    {
        // step 1: 定义你的数据结构

        // IrisData用于提供训练数据，并用作预测操作的输入
        // - 前4个属性是用来预测标签的输入或特征
        // - 标签是您预测的内容，并且仅在训练时设置
        public class IrisData
        {
            [Column("0")]
            public float SepalLength;   // 萼片长度

            [Column("1")]
            public float SepalWidth;    // 萼片宽度

            [Column("2")]
            public float PetalLength;   // 花瓣长度

            [Column("3")]
            public float PetalWidth;    // 花瓣宽度

            [Column("4")]
            [ColumnName("Label")]
            public string Label;        // 标签
        }

        // IrisPrediction是预测操作返回的结果
        public class IrisPrediction
        {
            [ColumnName("PredictedLabel")]
            public string PredictedLabels;  // 预测标签
        }

        static void Main(string[] args)
        {

            // Step 2: 创建管道读取您的数据
            var pipeline = new LearningPipeline();

            // 如果在Visual Studio中工作，请确保将iris-data.txt的“复制到输出目录”属性设置为“始终复制”
            string dataPath = "iris-data.txt";
            pipeline.Add(new TextLoader<IrisData>(dataPath, separator: ","));

            // Step 3: 转换您的数据将数值分配给“标签”列中的文本，因为在模型训练期间只能处理数字
            pipeline.Add(new Dictionarizer("Label"));

            // 将所有字段添加到vector中
            pipeline.Add(
                new ColumnConcatenator("Features", "SepalLength", "SepalWidth", "PetalLength", "PetalWidth"));

            // Step 4: 添加学习者
            // 向管道添加学习算法
            // 这是一种分类方案（这是什么类型的iris）
            pipeline.Add(new StochasticDualCoordinateAscentClassifier());

            // 将标签转换回原始文本（在步骤3转换为数字之后）
            pipeline.Add(new PredictedLabelColumnOriginalValueConverter() { PredictedLabelColumn = "PredictedLabel" });

            // Step 5: 根据数据训练您的模型
            var model = pipeline.Train<IrisData, IrisPrediction>();

            // Step 6: 使用您的模型来做出预测
            // 您可以改变这些数字来测试不同的预测结果
            var prediction = model.Predict(new IrisData()
            {
                SepalLength = 3.3f,
                SepalWidth = 1.6f,
                PetalLength = 0.2f,
                PetalWidth = 5.1f
            });

            Console.WriteLine($"预测花的类型为：{prediction.PredictedLabels}");
            Console.ReadKey();
        }
    }
}
