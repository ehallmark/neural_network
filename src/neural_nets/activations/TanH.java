package neural_nets.activations;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.ops.transforms.Transforms;

/**
 * Created by ehallmark on 11/6/16.
 */
public class TanH extends ActivationFunction {
    @Override
    public INDArray activation(INDArray in) {
        return Transforms.tanh(in).add(1.0).div(2);
    }

    @Override
    public INDArray derivative(INDArray in) {
        return Transforms.pow(Transforms.tanh(in),2).mul(-1.0).add(1.0).div(2.0);
    }
}
