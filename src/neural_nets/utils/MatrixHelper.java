package neural_nets.utils;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

/**
 * Created by ehallmark on 11/21/16.
 */
public class MatrixHelper {
    public static final double FUDGE_FACTOR = 0.00001;
    public static INDArray addBias(INDArray in) {
        return Nd4j.hstack(in, Nd4j.ones(in.rows(), 1));
    }
}
