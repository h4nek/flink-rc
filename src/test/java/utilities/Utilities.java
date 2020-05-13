package utilities;

import lm.streaming.ExampleStreamingUtilities;
import org.apache.flink.api.java.tuple.Tuple2;
import org.apache.flink.shaded.guava18.com.google.common.collect.Lists;

import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

/**
 * A general testing utilities class.
 */
public class Utilities {
    
    public static double[] listToArray(List<Double> input) {
        Double[] inputArr = (Double[]) input.toArray();
        double[] inputPrimitive = new double[inputArr.length];
        for (int i = 0; i < inputArr.length; i++) {
            inputPrimitive[i] = inputArr[i];
        }
        
        return inputPrimitive;
    }
    
    public static List<Double> arrayToList(double[] input) {
        Double[] inputObj = new Double[input.length];
        for (int i = 0; i < input.length; i++) {
            inputObj[i] = input[i];
        }
        List<Double> inputList;
        inputList = Lists.newArrayList(inputObj);
        
        return inputList;
    }

    public static <T> void write2DArrayToCSV(String pathToFile, Object[][] data) throws IOException {
        StringBuilder stringBuilder = new StringBuilder();
        for (int i = 0; i < data.length; ++i) {
            stringBuilder.append(data[i][0]);   // first element isn't preceded by a comma
            for (int j = 1; j < data[0].length; ++j) {  // put the row values in a comma-separated line
                stringBuilder.append(',').append(data[i][j]);
            }
            stringBuilder.append('\n');
        }
//        System.out.println("data prepared for CSV write: " + stringBuilder);
        
        File file = new File(pathToFile);
        file.getParentFile().mkdirs();
        file.createNewFile();
        FileWriter writer = new FileWriter(file);
        writer.write(stringBuilder.toString());
        writer.close();
    }

    /**
     * A utility function that converts a generic List of values into a String of comma-separated values.
     * Public for testing purposes (printing a list in code).
     * @param list
     * @param <T>
     * @return
     */
    public static <T> String listToString(List<T> list) {
        if (list == null)
            return "";

        StringBuilder listString = new StringBuilder();
        for (int i = 0; i < list.size(); ++i) {
            if (i == list.size() - 1) {
                listString.append(list.get(i));
            }
            else {
                listString.append(list.get(i)).append(",");
            }
        }

        return listString.toString();
    }

    /**
     * Creates a specified file and writes the contents of the specified list to it using common Java methods.
     * @param pathToFile
     * @param list
     * @param <T>
     * @throws IOException
     */
    public static <T> void writeListToFile(String pathToFile, List<T> list) throws IOException {
        File file = new File(pathToFile);
        file.getParentFile().mkdirs();
        file.createNewFile();
        FileWriter writer = new FileWriter(file);
        writer.write(Utilities.listToString(list) + '\n');
        writer.close();
    }
}
