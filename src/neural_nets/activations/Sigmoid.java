package neural_nets.activations;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.ops.transforms.Transforms;

/**
 * Created by ehallmark on 11/6/16.
 */
public class Sigmoid extends ActivationFunction {
    @Override
    public INDArray activation(INDArray in) {
        return Transforms.sigmoid(in);
    }

    @Override
    public INDArray derivative(INDArray in) {
        INDArray sigmoid = activation(in);
        return sigmoid.mul(sigmoid.mul(-1.0).add(1.0));
    }
}
