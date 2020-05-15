package higher_level_examples;

import org.apache.flink.api.common.functions.FlatMapFunction;
import org.apache.flink.util.Collector;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;

/**
 * An input processing function, common for all HLEs.
 * Accepts lines of CSV file as {@code String} values. Converts each into a vector ({@code List<Double>}), 
 * possibly using custom parsers.
 * <br><br>
 * The vector is later broken into input, output and input for plotting parts. If we want one column to be used for 
 * both input and output, we can duplicate it through the custom parser.
 */
public class ProcessInput implements FlatMapFunction<String, List<Double>> {
    private String columnsBitMask;
    private Map<Integer, DataParsing> customParsers;
    private boolean debugging;

    public ProcessInput(String columnsBitMask, Map<Integer, DataParsing> customParsers, boolean debugging) {
        this.columnsBitMask = columnsBitMask;
        this.customParsers = customParsers;
        this.debugging = debugging;
    }

    @Override
    public void flatMap(String line, Collector<List<Double>> out) throws Exception {
        String[] items = line.split(",");
        List<Double> inputVector = new ArrayList<>();
        for (int i = 0; i < items.length; ++i) {
            // perform any necessary modifications
            // (e.g. "normalize" the data to be in some reasonable range for the transformation)
            if (columnsBitMask.charAt(i) != '0') {
                try {
                    if (customParsers != null && customParsers.containsKey(i)) {
                        customParsers.get(i).parseAndAddData(items[i], inputVector);
                    } else {
                        inputVector.add(Double.parseDouble(items[i]));
                    }
                }
                catch (Exception e) {   // dealing with invalid/unwanted lines - exclude them
                    if (debugging) {
                        System.err.println("invalid cell: " + items[i]);
                        System.err.println("line: " + line);
                    }
                    return;  // we don't want to process other cells
                }
            }
        }
        out.collect(inputVector); // the line is valid
    }
}
