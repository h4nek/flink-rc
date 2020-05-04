package rc;

import higher_level_examples.HigherLevelExampleAbstract;
import higher_level_examples.HigherLevelExampleBatch;
import higher_level_examples.DataParsing;

import java.util.List;

public class GlacierMeltdownExample {
    public static final String INPUT_FILE_PATH = "src/test/resources/glaciers/input_data/glaciers.csv";
    private static final int N_u = 1;
    private static final int N_x = 6;
    private static final double learningRate = 0.01;
    
    public static void main(String[] args) throws Exception {
        HigherLevelExampleAbstract.setup(INPUT_FILE_PATH, "110", 1, N_u, N_x, false, 
                null, true, (int) Math.floor(0.8*69), learningRate, true,
                0);
        HigherLevelExampleAbstract.addCustomParser(0, (inputString, inputVector) -> {
            double year = Double.parseDouble(inputString);
            inputVector.add(0, (year - 1945 - 34)/34);    // move the column values to be around 0
        });
//        HigherLevelExampleAbstract.addCustomParser(1, (x, y) -> {double mwe = Double.parseDouble(x); y.add((mwe -14)/14);});
        HigherLevelExampleAbstract.addCustomParser(2, (x, y) -> {double observations = Double.parseDouble(x); y.add((observations - 18)/18);});
//        HigherLevelExampleAbstract.setupPlotting();
        HigherLevelExampleAbstract.addPlottingTransformer(0, x -> x*34 + 1945 + 34);
//        HigherLevelExampleAbstract.addPlottingTransformer(1, x -> x*18 + 18);
        HigherLevelExampleBatch.run();
    }
}
