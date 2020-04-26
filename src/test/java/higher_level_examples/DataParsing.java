package higher_level_examples;

import java.util.List;

public interface DataParsing {
    /**
     * Converts the String data elements of a selected column into one (, zero) or possibly multiple double values, each 
     * typically representing a feature of the input vector u(t) or the output scalar y(t).
     * @param inputString a value from the chosen column of a CSV file, supplied as a String
     * @param inputVector a reference to the joint input/output vector where all values (features) are expected to be 
     *                    added
     */
    void parseAndAddInput(String inputString, List<Double> inputVector);
}
