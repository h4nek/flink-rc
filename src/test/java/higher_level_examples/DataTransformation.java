package higher_level_examples;

import java.io.Serializable;

/**
 * Used for Python Plotting
 */
public interface DataTransformation extends Serializable {
    double transform(double input);
}
