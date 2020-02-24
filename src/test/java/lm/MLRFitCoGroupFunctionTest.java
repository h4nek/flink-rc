package lm;

import Jama.Matrix;
import com.sun.javaws.exceptions.InvalidArgumentException;
import org.apache.flink.api.common.functions.util.ListCollector;
import org.apache.flink.api.java.tuple.Tuple2;
import org.apache.flink.util.Collector;
import org.junit.jupiter.api.*;

import java.util.*;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

import static org.junit.jupiter.api.Assertions.*;

class MLRFitCoGroupFunctionTest {

    /**
     * The simplest test using identity function (f(x) = x).
     * Making a small change to Alpha parameter and expecting a small change of the new Alpha.
     * 
     * Computing "finite differences" as advised in the book Neural Networks: Tricks of the Trade, page 429
     * Reference: http://cs231n.github.io/neural-networks-3/#gradcheck
     */
    @Test
    void trainUsingGradientDescent() throws InvalidArgumentException {
        double LEARNING_RATE = 0.01;
        int numSamples = 100;
        // create a list of integers from 1 to 100, converted to Double type
        List<List<Double>> inputList = IntStream.rangeClosed(1, numSamples).boxed()
                .map(x -> Arrays.asList(x.doubleValue(), x.doubleValue())).collect(Collectors.toList());
        Random random = new Random();
        List<List<Double>> alphaList = new ArrayList<>();
        List<List<Double>> alphaAltList = new ArrayList<>();
        for (int i = 0; i < numSamples; i++) {
            List<Double> alpha = new ArrayList<>();
            List<Double> alphaAlt = new ArrayList<>();

            alpha.add(random.nextDouble()*350);
            alpha.add(random.nextDouble()*800);
            alphaList.add(alpha);

            alphaAlt.add(alpha.get(0) + random.nextDouble()*1e-3);
            alphaAlt.add(alpha.get(1) + random.nextDouble()*1e-3);
            alphaAltList.add(alphaAlt);
        }
        List<Double> outputList = IntStream.rangeClosed(1, numSamples).boxed().map(Integer::doubleValue).collect(Collectors.toList());

        List<Tuple2<Long, List<Double>>> outList = new ArrayList<>();
        Collector<Tuple2<Long, List<Double>>> out = new ListCollector<Tuple2<Long, List<Double>>>(outList);
        List<Tuple2<Long, List<Double>>> outAltList = new ArrayList<>();
        Collector<Tuple2<Long, List<Double>>> outAlt = new ListCollector<Tuple2<Long, List<Double>>>(outAltList);
        for (int i = 0; i < 100; i++) {
            MLRFitCoGroupFunction mlrFit = new MLRFitCoGroupFunction(alphaList.get(i), LEARNING_RATE, numSamples, 
                    true, false, Double.NaN, Double.NaN);
            MLRFitCoGroupFunction mlrFitAlt = new MLRFitCoGroupFunction(alphaList.get(i), LEARNING_RATE, numSamples,
                    false, false, Double.NaN, Double.NaN);   // used to make sure the two 
            // computations don't interfere (separate MSE)
            System.out.println("GD Arguments:\nx=" + inputList.get(i) + "\talpha=" + alphaList.get(i) + "\ty=" + outputList.get(i));
            List<Double> alphaNew = mlrFit.trainUsingGradientDescent(alphaList.get(i), inputList.get(i), 
                    outputList.get(i), LEARNING_RATE, numSamples, out);
            List<Double> alphaAltNew = mlrFitAlt.trainUsingGradientDescent(alphaAltList.get(i), inputList.get(i), 
                    outputList.get(i), LEARNING_RATE, numSamples, outAlt);
            
            /*Computing GD should result in small change of Alpha*/
            System.out.println("Computed Alpha=" + alphaNew + "\tOld Alpha=" + alphaList.get(i));
            assertEquals(alphaList.get(i).get(0), alphaNew.get(0), 0.2*inputList.get(i).get(0)*inputList.get(i).get(0));
            assertEquals(alphaList.get(i).get(1), alphaNew.get(1), 0.2*inputList.get(i).get(1)*inputList.get(i).get(1));

            /*Small change of Alpha that we input to GD should result in small change of output Alpha*/
            System.out.println("Computed Alpha 0=" + alphaNew.get(0) + "\tComputed Alt Alpha 0=" + alphaAltNew.get(0));
            System.out.println("Computed Alpha 1=" + alphaNew.get(1) + "\tComputed Alt Alpha 1=" + alphaAltNew.get(1));
            assertEquals(alphaNew.get(0), alphaAltNew.get(0), 0.01);
            assertEquals(alphaNew.get(1), alphaAltNew.get(1), 0.01);
            
            /*Finite differences ?*/
            //  gradient ~= (f(x+h) - f(x-h)) / 2h
            
            
            /*Check MSE*/
            double mseMLRFit = outList.get(i).f1.get(0);
            double mseLocal = mse(inputList.get(i), alphaList.get(i), outputList.get(i), numSamples);
            System.out.println("MLRFit MSE: " + mseMLRFit + "\tTest MSE: " + mseLocal);
            assertEquals(mseMLRFit, mseLocal);
        }
    }

    double mse(List<Double> x, List<Double> alpha, double y_real, int numSamples) {
        double y_pred = 0;
        for (int i = 0; i < alpha.size(); i++) {
            y_pred += x.get(i)*alpha.get(i);
        }
        return Math.pow(y_pred - y_real, 2)/numSamples;
    }
}
