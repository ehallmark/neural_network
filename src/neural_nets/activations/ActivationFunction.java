package neural_nets.activations;

import neural_nets.layers.Layer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.NDArrayIndex;

import java.io.Serializable;

/**
 * Created by ehallmark on 11/6/16.
 */
public abstract class ActivationFunction implements Serializable {
    public abstract INDArray activation(INDArray in);
    public abstract INDArray derivative(INDArray in);
    // default gradient procedure
    public INDArray computeGradient(Layer layer, INDArray expectedOutputs, INDArray previousDelta, INDArray previousWeights) {
        INDArray delta;
        if(previousDelta!=null&&previousWeights!=null) {
            delta = previousDelta.mmul(previousWeights.get(NDArrayIndex.interval(0, previousWeights.rows() - 1, false), NDArrayIndex.all()).transposei());
        } else {
            // Top Layer
            delta = expectedOutputs.sub(layer.getOutputs());
        }
        //System.out.println("delta: "+delta.toString());
        //System.out.println("deriv: "+derivative(layer.nets).toString());
        if(layer.getActivation() instanceof Sigmoid) {
            delta.muli(Nd4j.ones(layer.getOutputs().shape()).subi(layer.getOutputs()).muli(layer.getOutputs()));
        } else if(!(layer.getActivation() instanceof SoftMax)) {
            delta.muli(derivative(layer.getNets()));
        }
        return delta;
    }
}
