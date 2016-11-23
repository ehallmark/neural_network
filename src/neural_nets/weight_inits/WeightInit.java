package neural_nets.weight_inits;

import org.nd4j.linalg.api.ndarray.INDArray;

import java.io.Serializable;

/**
 * Created by ehallmark on 11/8/16.
 */
public interface WeightInit extends Serializable {
    INDArray getWeightInits(int rows, int cols);
}
