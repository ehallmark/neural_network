package neural_nets.layers;

import edu.stanford.nlp.util.Pair;
import neural_nets.activations.ActivationFunction;
import neural_nets.models.NeuralNetwork;
import neural_nets.weight_inits.WeightInit;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.INDArrayIndex;
import org.nd4j.linalg.indexing.NDArrayIndex;
import org.nd4j.linalg.ops.transforms.Transforms;

/**
 * Created by ehallmark on 11/21/16.
 */
public interface Layer {
    INDArray getOutputs();

    INDArray getInputs();

    INDArray getNets();

    void updateWeights();

    ActivationFunction getActivation();

    void setDelta(INDArray delta);

    INDArray backProp(INDArray previousWeights, INDArray previousDelta, INDArray targetOutput);

    Pair<INDArray, INDArray> feedForward(INDArray inputs, boolean train);

    INDArray getSampleNetOutputs(INDArray weights, INDArray inputsWithBias);

    INDArray getWeights();

    INDArray getWeights(boolean train);

    int getInputCount();

    int getOutputCount();

    void initLayer(NeuralNetwork net, boolean adaGrad, double learningRate, double momentum, WeightInit weightInit, ActivationFunction function);
}
