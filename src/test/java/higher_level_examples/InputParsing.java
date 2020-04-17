package higher_level_examples;

import java.util.List;

public interface InputParsing {
    /**
     * Converts the String input of a selected column into one (, zero) or possibly multiple double values, each 
     * representing a feature of the input vector u(t).
     * @param inputString a value from the chosen column of a CSV file, supplied as a String
     * @param inputVector A reference to the vector u(t) where additional values (features) are expected to be added
     */
    void parseAndAddInput(String inputString, List<Double> inputVector);
}
