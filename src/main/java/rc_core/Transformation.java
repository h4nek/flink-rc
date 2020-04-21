package rc_core;

import java.io.Serializable;

public interface Transformation extends Serializable {
    double transform(double d);
}
