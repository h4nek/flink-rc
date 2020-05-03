package higher_level_examples;

import lm.LinearRegression;
import lm.LinearRegressionPrimitive;
import lm.batch.ExampleBatchUtilities;
import org.apache.flink.api.common.typeinfo.Types;
import org.apache.flink.api.java.DataSet;
import org.apache.flink.api.java.ExecutionEnvironment;
import org.apache.flink.api.java.io.TextInputFormat;
import org.apache.flink.api.java.tuple.Tuple2;
import org.apache.flink.core.fs.Path;
import rc_core.ESNReservoirSparse;
import utilities.BasicIndexer;
import utilities.PythonPlotting;
import utilities.Utilities;

import java.util.*;

public class HigherLevelExampleBatch extends HigherLevelExampleAbstract {
    public static void main(String[] args) throws Exception {
        ExecutionEnvironment env = ExecutionEnvironment.getExecutionEnvironment();
        env.setParallelism(1);

        DataSet<List<Double>> dataSet = env.readFile(new TextInputFormat(new Path(inputFilePath)), inputFilePath)
                .flatMap(new ProcessInput()).returns(Types.LIST(Types.DOUBLE));
        if (debugging) dataSet.printOnTaskManager("DATA"); //TEST

        DataSet<Tuple2<Long, List<Double>>> indexedDataSet = dataSet.map(new BasicIndexer<>());
        if (debugging) indexedDataSet.printOnTaskManager("INDEXED DATA");  //TEST
        
        DataSet<Tuple2<Long, List<Double>>> inputSet = indexedDataSet.map(x -> {x.f1.remove(outputIdx); return x;})
                .returns(Types.TUPLE(Types.LONG, Types.LIST(Types.DOUBLE)));
        DataSet<Tuple2<Long, Double>> outputSet = indexedDataSet.map(x -> Tuple2.of(x.f0, x.f1.get(outputIdx)))
                .returns(Types.TUPLE(Types.LONG, Types.DOUBLE));
        if (debugging) inputSet.printOnTaskManager("IN");
        if (debugging) outputSet.printOnTaskManager("OUT");
        
        DataSet<Tuple2<Long, List<Double>>> reservoirOutput = inputSet.map(new ESNReservoirSparse(N_u, N_x, 
                init_vector, transformation, range, shift, jumpSize, sparsity, scalingAlpha,
                reservoirTopology, includeInput, includeBias));
        if (debugging) reservoirOutput.printOnTaskManager("Reservoir output");

        DataSet<Tuple2<Long, List<Double>>> trainingInput = reservoirOutput.first(trainingSetSize);
        DataSet<Tuple2<Long, Double>> trainingOutput = outputSet.first(trainingSetSize);
        DataSet<Tuple2<Long, List<Double>>> testingInput = reservoirOutput.filter(x -> x.f0 >= trainingSetSize);
        DataSet<Tuple2<Long, Double>> testingOutput = outputSet.filter(x -> x.f0 >= trainingSetSize);
        
        LinearRegression lr = new LinearRegression();
        DataSet<Tuple2<Long, List<Double>>> alphas = lr.fit(trainingInput, trainingOutput, lmAlphaInit,
                learningRate, trainingSetSize, includeMSE, stepsDecay);
        if (includeMSE) {
            DataSet<Tuple2<Long, List<Double>>> MSEs = alphas.filter(x -> x.f0 == -1);
            alphas = alphas.filter(x -> x.f0 != -1);
            if (debugging) MSEs.printOnTaskManager("MSE");
        }
        if (debugging) alphas.printOnTaskManager("ALPHA");

        List<List<Double>> alphaList = alphas.map(x -> x.f1).returns(Types.LIST(Types.DOUBLE)).collect();
        List<Double> finalAlpha = alphaList.get(alphaList.size() - 1);
        if (debugging) System.out.println("Final Alpha: " + Utilities.listToString(finalAlpha));
        
        DataSet<Tuple2<Long, Double>> predictions = LinearRegressionPrimitive.predict(testingInput, finalAlpha);

        /* Do the offline (pseudoinverse) fitting for comparison */
        List<Double> AlphaOffline = LinearRegressionPrimitive.fit(trainingInput, trainingOutput, 
                LinearRegressionPrimitive.TrainingMethod.PSEUDOINVERSE, regularizationFactor);
        if (debugging) System.out.println("Offline Alpha: " + AlphaOffline);
        DataSet<Tuple2<Long, Double>> predictionsOffline = LinearRegressionPrimitive.predict(testingInput, AlphaOffline);
        
        ExampleBatchUtilities.computeAndPrintOfflineOnlineMSE(predictionsOffline, predictions, testingOutput);
        
        if (debugging)  // format: (index, [input, output, prediction, offline prediction])
            indexedDataSet.join(predictions).where(0).equalTo(0)
                .with((x,y) -> {List<Double> inputOutput = x.f1; inputOutput.add(y.f1); return Tuple2.of(x.f0, inputOutput);})
                .returns(Types.TUPLE(Types.LONG, Types.LIST(Types.DOUBLE)))
                .join(predictionsOffline).where(0).equalTo(0)
                .with((x,y) -> {List<Double> results = x.f1; results.add(y.f1); return Tuple2.of(x.f0, results);})
                .returns(Types.TUPLE(Types.LONG, Types.LIST(Types.DOUBLE)))
                .printOnTaskManager("RESULTS");


        DataSet<Tuple2<Long, List<Double>>> plottingInputSet = inputSet.filter(x -> x.f0 >= trainingSetSize)
                .map(x -> { 
                    for (int i = 0; i < x.f1.size(); ++i) {
                        if (plottingTransformers.containsKey(i)) {
                            Double transformed = plottingTransformers.get(i).transform(x.f1.remove(i));
                            x.f1.add(i, transformed); 
                        } 
                    }
                    return x; 
                }).returns(Types.TUPLE(Types.LONG, Types.LIST(Types.DOUBLE)));
        List<Tuple2<Long, Double>> plottingOutputSet = modifyForPlotting(outputSet);
        List<Tuple2<Long, Double>> plottingPredictions = modifyForPlotting(predictions);
        List<Tuple2<Long, Double>> plottingPredictionsOffline = modifyForPlotting(predictionsOffline);
        PythonPlotting.plotRCPredictions(plottingInputSet.collect(), plottingOutputSet, plottingPredictions,
                plotFileName, xlabel, ylabel, title, inputIndex, shiftData, plotType, inputHeaders, outputHeaders, 
                plottingPredictionsOffline);
    }

    /**
     * Used for invoking the example through another method.
     */
    public static void run() throws Exception {
        main(null);
    }

    private static List<Tuple2<Long, Double>> modifyForPlotting(DataSet<Tuple2<Long, Double>> dataSet) throws Exception {
        return dataSet.filter(x -> x.f0 >= trainingSetSize).map(y -> {
            if (plottingTransformers.containsKey(N_u)) {
                y.f1 = plottingTransformers.get(N_u).transform(y.f1);
            }
            return y;
        }).returns(Types.TUPLE(Types.LONG, Types.DOUBLE)).collect();
    }
}
