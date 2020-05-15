package higher_level_examples;

import java.io.Serializable;
import java.util.List;

public interface DataParsing extends Serializable {
    /**
     * Converts the String data elements of a selected column into one (, zero) or possibly multiple double values, each 
     * typically representing a feature of the input vector u(t) or the output scalar y(t).
     * @param inputString a value from the chosen column of a CSV file, supplied as a String
     * @param inputVector a reference to the joint input/output vector where all values (features) are expected to be 
     *                    added. the order should be: input fields | input feature for plotting | output field
     * @throws Exception can be any exception thrown to exclude the line from the dataset; or if the current value 
     *                   can't be parsed
     */
    void parseAndAddData(String inputString, List<Double> inputVector) throws Exception;
}
