package lm.batch;

import lm.LinearRegressionPrimitive;
import lm.streaming.ExampleStreamingUtilities;
import org.apache.flink.api.common.functions.JoinFunction;
import org.apache.flink.api.common.functions.MapFunction;
import org.apache.flink.api.common.functions.RichMapFunction;
import org.apache.flink.api.java.DataSet;
import org.apache.flink.api.java.tuple.Tuple2;
import org.apache.flink.api.java.tuple.Tuple3;
import org.apache.flink.configuration.Configuration;
import org.apache.flink.metrics.Counter;
import org.math.plot.Plot2DPanel;
import org.math.plot.PlotPanel;

import javax.swing.*;
import java.awt.*;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

public class ExampleBatchUtilities {
    
    public static <T> void writeListDataSetToFile(String pathToFile, List<List<T>> list, List<String> headers) throws IOException {
        File file = new File(pathToFile);
        file.getParentFile().mkdirs();
        file.createNewFile();
        FileWriter writer = new FileWriter(file);
        writer.write(ExampleStreamingUtilities.listToString(headers) + '\n');
        for (int i = 0; i < list.size(); ++i) {
            List<T> elem = list.get(i);
            writer.write(i + "," + ExampleStreamingUtilities.listToString(elem) + '\n');
        }
        writer.close();
    }
    
    public static <T> void writeInputDataSetToFile(String pathToFile, List<Tuple2<Long, List<T>>> list) throws IOException {
        File file = new File(pathToFile);
        file.getParentFile().mkdirs();
        file.createNewFile();
        FileWriter writer = new FileWriter(file);
        for (int i = 0; i < list.size(); ++i) {
            Tuple2<Long, List<T>> tuple = list.get(i);
            
            writer.write(tuple.f0 + "," + ExampleStreamingUtilities.listToString(tuple.f1) + '\n');
        }
        writer.close();
    }
    
    public static <T> void writeOutputDataSetToFile(String pathToFile, List<Tuple2<Long, T>> list) throws IOException {
        File file = new File(pathToFile);
        file.getParentFile().mkdirs();
        file.createNewFile();
        FileWriter writer = new FileWriter(file);
        for (int i = 0; i < list.size(); ++i) {
            Tuple2<Long, T> tuple = list.get(i);

            writer.write(tuple.f0 + "," + tuple.f1 + '\n');
        }
        writer.close();
    }
    
    public static void computeAndPrintOfflineOnlineMSE(DataSet<Tuple2<Long, Double>> predictionsOffline, 
                                                       DataSet<Tuple2<Long, Double>> predictionsOnline,
                                                       DataSet<Tuple2<Long, Double>> outputSet) throws Exception {
        List<Double> mseOfflineList = computeMSE(predictionsOffline, outputSet).collect();
        List<Double> mseOnlineList = computeMSE(predictionsOnline, outputSet).collect();

        System.out.println("MSE offline: " + mseOfflineList.get(mseOfflineList.size() - 1));
        System.out.println("MSE online:  " + mseOnlineList.get(mseOnlineList.size() - 1));
    }

    public static DataSet<Double> computeMSE(DataSet<Tuple2<Long, Double>> predictions, DataSet<Tuple2<Long, Double>> outputSet) {

        /* Join the stream of predictions with the stream of real outputs */
        DataSet<Tuple3<Long, Double, Double>> predsAndReal = predictions.join(outputSet).where(x -> x.f0).equalTo(y -> y.f0)
                .with(
                        new JoinFunction<Tuple2<Long, Double>, Tuple2<Long, Double>, Tuple3<Long, Double, Double>>() {
                            @Override
                            public Tuple3<Long, Double, Double> join(Tuple2<Long, Double> y_pred, Tuple2<Long, Double> y) throws Exception {
                                return Tuple3.of(y_pred.f0, y_pred.f1, y.f1);
                            }
                        });

//        predsAndReal.printOnTaskManager("PREDS AND REAL");

        /* Compare the results - compute the MSE */
        DataSet<Double> mse = predsAndReal.map(new MapFunction<Tuple3<Long, Double, Double>, Double>() {
            private double MSESum = 0;
            private int numRecords = 0;

            @Override
            public Double map(Tuple3<Long, Double, Double> input) throws Exception {
                MSESum += Math.pow(input.f1 - input.f2, 2);
                ++numRecords;
                
                return MSESum / numRecords;
            }
        });

        return mse;
    }
    
    public static void showMSETrend(List<List<Double>> alphaList, DataSet<Tuple2<Long, List<Double>>> inputSet,
                                    DataSet<Tuple2<Long, Double>> outputSet) throws Exception {
        List<Double> mseTrend = computeMSETrend(alphaList, inputSet, outputSet);
        for (int i = 0; i < mseTrend.size(); ++i) {
            System.out.println(i + ". MSE: " + mseTrend.get(i));
        }
    }
    
    public static List<Double> computeMSETrend(List<List<Double>> alphaList, DataSet<Tuple2<Long, List<Double>>> inputSet,
                                       DataSet<Tuple2<Long, Double>> outputSet) throws Exception {
        List<Double> mseTrend = new ArrayList<>();
        for (int i = 0; i < alphaList.size(); i++) {
            List<Double> curAlpha = alphaList.get(i);
            DataSet<Tuple2<Long, Double>> predictions = LinearRegressionPrimitive.predict(inputSet, curAlpha);
            
            double mse = ExampleBatchUtilities.computeMSE(predictions, outputSet).collect().get(alphaList.size() - 1);
            System.out.println(i + ". MSE: " + mse);    //TEST
            mseTrend.add(mse);
        }
        return mseTrend;
    }

    public static List<Double> selectMinAlpha(List<List<Double>> alphaList, DataSet<Tuple2<Long, List<Double>>> inputSet,
                                              DataSet<Tuple2<Long, Double>> outputSet) throws Exception  {
        List<Double> minAlpha = null;
        double minErr = Double.POSITIVE_INFINITY;
        List<Tuple2<Long, List<Double>>> inputList = inputSet.collect();
        List<Tuple2<Long, Double>> outputList = outputSet.collect();
        for (int i = 0; i < alphaList.size(); i++) {
            List<Double> curAlpha = alphaList.get(i);
            /*Compute MSE just for one sample*/
            List<Double> inputVector = inputList.get(i).f1;
            inputVector.add(0, 1.0); // add an extra value for the intercept

            double y_pred = 0;
            for (int j = 0; j < curAlpha.size(); j++) {
                y_pred += curAlpha.get(j) * inputVector.get(j);
            }
            double y_real = outputList.get(i).f1;

            double err = Math.pow(y_pred - y_real, 2)/inputList.size();
            if (err < minErr) {
                minAlpha = curAlpha;
                minErr = err;
            }
        }
        return minAlpha;
    }

    public enum PlotType {
        LINE,
        POINTS
    }
    
    private JFrame frame; // keeps the current state of the plot


    public void plotLRFit(DataSet<Tuple2<Long, List<Double>>> inputSet, DataSet<Tuple2<Long, Double>> outputSet,
                                 DataSet<Tuple2<Long, Double>> predictions, int inputIndex) throws Exception {
        plotLRFit(inputSet, outputSet, predictions, inputIndex, 0, "x", "y", "Linear Regression", 
                PlotType.POINTS);
    }
    
    /**
     * Using JMathPlot (https://github.com/yannrichet/jmathplot) to plot the LR input and output relation graph.
     * @param inputSet
     * @param outputSet
     * @param inputIndex Index of the input variable that we want to plot (in the Doubles List).
     * @throws Exception
     */
    public void plotLRFit(DataSet<Tuple2<Long, List<Double>>> inputSet, DataSet<Tuple2<Long, Double>> outputSet,
                                 DataSet<Tuple2<Long, Double>> predictions, int inputIndex, int shiftData, String xlabel, String ylabel,
                                 String title, PlotType plotType)
            throws Exception {
        List<Tuple2<Long, List<Double>>> inputList = inputSet.collect();
        List<Tuple2<Long, Double>> outputList = outputSet.collect();
        List<Tuple2<Long, Double>> predictionsList = predictions.collect();
        double[] inputArr = new double[inputList.size()];
        double[] outputArr = new double[outputList.size()];
        double[] inputPredArr = new double[predictionsList.size()];
        double[] predArr = new double[predictionsList.size()];

        for (int i = 0; i < inputList.size(); ++i) {
            Tuple2<Long, List<Double>> input = inputList.get(i);
            Tuple2<Long, Double> output = outputList.get(i);
            Tuple2<Long, Double> prediction = predictionsList.get(i);
            
            inputArr[i] = input.f1.get(inputIndex);
            outputArr[i] = output.f1;
            
            inputPredArr[i] = input.f1.get(inputIndex) + shiftData;
            predArr[i] = prediction.f1;
        }

        // create your PlotPanel (you can use it as a JPanel)
        Plot2DPanel plot = new Plot2DPanel();
        // add line/scatter plots to the PlotPanel
        if (plotType == PlotType.POINTS) {
            plot.addScatterPlot("original output", inputArr, outputArr);
        }
        else {
            plot.addLinePlot("original output", inputArr, outputArr);
        }
        plot.addLinePlot("LR fit", Color.RED, inputArr, predArr);
        plot.addLegend(PlotPanel.NORTH);
        plot.getAxis(0).setLabelText(xlabel);
        plot.getAxis(1).setLabelText(ylabel);

        // put the PlotPanel in a JFrame, as a JPanel
        frame = new JFrame(title);
        frame.setContentPane(plot);
        frame.setMinimumSize(new Dimension(800, 600));
        frame.toFront();
        frame.setVisible(true);
    }
    
    public void addLRFitToPlot(DataSet<Tuple2<Long, List<Double>>> inputSet, DataSet<Tuple2<Long, Double>> predictions,
                               int inputIndex) throws Exception {
        addLRFitToPlot(inputSet, predictions, inputIndex, 0, "Pseudoinverse LR Fit", Color.GREEN);
    }

    /**
     * Add a LR fit to the existing plot (stored as a field of this class).
     */
    public void addLRFitToPlot(DataSet<Tuple2<Long, List<Double>>> inputSet, DataSet<Tuple2<Long, Double>> predictions, 
                               int inputIndex, int shiftData, String name, Color color) throws Exception {
        List<Tuple2<Long, List<Double>>> inputList = inputSet.collect();
        List<Tuple2<Long, Double>> predictionsList = predictions.collect();
        double[] inputArr = new double[inputList.size()];
        double[] inputPredArr = new double[predictionsList.size()];
        double[] predArr = new double[predictionsList.size()];

        for (int i = 0; i < inputList.size(); ++i) {
            Tuple2<Long, List<Double>> input = inputList.get(i);
            Tuple2<Long, Double> prediction = predictionsList.get(i);

            inputArr[i] = input.f1.get(inputIndex);

            inputPredArr[i] = input.f1.get(inputIndex) + shiftData;
            predArr[i] = prediction.f1;
        }
        Plot2DPanel plot = (Plot2DPanel) frame.getContentPane();
        plot.addLinePlot(name, color, inputArr, predArr);
        frame.setContentPane(plot); // the JFrame is automatically refreshed
    }

    /**
     * Plot the trend of MSE associated with the Alpha vector state at the time.
     * Shows us how well (in our measures) the model was trained throughout fitting.
     * @param mseTrend
     */
    public static void plotLearningCurve(List<Double> mseTrend) {
        double[] mseArr = new double[mseTrend.size()];
        for (int i = 0; i < mseTrend.size(); i++) {
            mseArr[i] = mseTrend.get(i);
        }

        // create your PlotPanel (you can use it as a JPanel)
        Plot2DPanel plot = new Plot2DPanel();
        // add line plot to the PlotPanel
        plot.addLinePlot("MSE Trend", mseArr);
        plot.getAxis(0).setLabelText("Index");
        plot.getAxis(1).setLabelText("MSE");

        // put the PlotPanel in a JFrame, as a JPanel
        JFrame frame = new JFrame("LR Learning Curve");
        frame.setContentPane(plot);
        frame.setMinimumSize(new Dimension(800, 600));
        frame.toFront();
        frame.setVisible(true);
    }

    /**
     * Plot a single Alpha vector value (scalar) in relation to the number of iterations.
     */
    public static void plotAlphaParameter(List<Double> alphaInTime, String suffix) {
        double[] alphaArr = new double[alphaInTime.size()];
        for (int i = 0; i < alphaInTime.size(); i++) {
            alphaArr[i] = alphaInTime.get(i);
        }

        // create your PlotPanel (you can use it as a JPanel)
        Plot2DPanel plot = new Plot2DPanel();
        // add line plot to the PlotPanel
        plot.addLinePlot("Alpha" + suffix + " Trend", alphaArr);
        plot.getAxis(0).setLabelText("Iteration");
        plot.getAxis(1).setLabelText("Alpha " + suffix);

        // put the PlotPanel in a JFrame, as a JPanel
        JFrame frame = new JFrame("Alpha " + suffix + " parameter training");
        frame.setContentPane(plot);
        frame.setMinimumSize(new Dimension(800, 600));
        frame.toFront();
        frame.setVisible(true);
    }

    /**
     * Plot training trend of each coordinate of the Alpha vector 
     * Default suffix (0-based index) will be used for individual coordinates.
     * @param alphaList
     */
    public static void plotAllAlphas(List<List<Double>> alphaList) {
        List<Double> alphaInTime = new ArrayList<>();
        for (int j = 0; j < alphaList.get(0).size(); ++j) {  // iterate over all alpha values from i-th step
            for (int i = 0; i < alphaList.size(); ++i) {    // iterate over alphas from all training steps
                alphaInTime.add(alphaList.get(i).get(j));
            }
            ExampleBatchUtilities.plotAlphaParameter(alphaInTime, "" + j);
            alphaInTime.clear();
        }
    }

    /**
     * Index the elements of the dataset/datastream.
     * @param <T>
     */
    public static class IndicesMapper<T> extends RichMapFunction<T, Tuple2<Long, T>> {
        private Counter counter;

        @Override
        public void open(Configuration parameters) throws Exception {
            super.open(parameters);

            counter = getRuntimeContext().getMetricGroup().counter("index counter");
        }

        @Override
        public Tuple2<Long, T> map(T value) throws Exception {

            Tuple2<Long, T> indexedValue = Tuple2.of(counter.getCount(), value);
            counter.inc();
            return indexedValue;
        }
    }
}
