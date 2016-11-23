package neural_nets.activations;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.ops.transforms.Transforms;

/**
 * Created by ehallmark on 11/6/16.
 */
public class ReLU extends ActivationFunction {
    @Override
    public INDArray activation(INDArray in) {
        return Transforms.relu(in);
    }

    @Override
    public INDArray derivative(INDArray in) {
        return Transforms.sign(in).add(1.0).div(2.0);
    }
}
