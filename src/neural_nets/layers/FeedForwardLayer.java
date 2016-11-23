package neural_nets.layers;

import edu.stanford.nlp.util.Pair;
import neural_nets.activations.ActivationFunction;
import neural_nets.activations.Sigmoid;
import neural_nets.models.NeuralNetwork;
import neural_nets.utils.BernoulliRandom;
import neural_nets.utils.MatrixHelper;
import neural_nets.weight_inits.WeightInit;
import neural_nets.weight_inits.XavierFanAvg;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.INDArrayIndex;
import org.nd4j.linalg.indexing.NDArrayIndex;
import org.nd4j.linalg.ops.transforms.Transforms;

import java.io.Serializable;
import java.util.*;

public class FeedForwardLayer implements Layer,Serializable {
    protected int outputCount;
    protected int inputCount;
    protected ActivationFunction function = new Sigmoid();
    private INDArray weights;
    protected transient INDArray nets;
    protected double negativeSampling = -1.0;
    protected transient INDArray outputs;
    protected transient INDArray inputs;
    protected double learningRate;
    protected INDArray lastDeltaWeights;
    protected INDArray delta;
    protected WeightInit weightInit = new XavierFanAvg();
    protected double maxNormRegularizationConstant = -1.0;
    protected double momentum = 0.0;
    protected boolean adaGrad = false;
    protected NeuralNetwork network;
    protected INDArray historicalDelta;
    protected double dropConnect = -1;
    protected int numSamples = 30;
    private transient INDArray dropConnectWeights;
    public boolean adaGradWasSet = false;
    public boolean learningRateWasSet = false;
    public boolean weightInitWasSet = false;
    public boolean momentumWasSet = false;
    public boolean activationWasSet = false;
    protected static Random rand = new Random(System.currentTimeMillis());

    public FeedForwardLayer(int inputCount, int outputCount) {
        this.inputCount = inputCount;
        this.outputCount = outputCount;
        this.weights = weightInit.getWeightInits(inputCount + 1, outputCount);
        regularizeWeights();
    }

    public static INDArray randomBernoulliVariables(int[] shape, double p) {
        return new BernoulliRandom(p).nextDouble(shape);
    }

    public void regularizeWeights() {
        if(maxNormRegularizationConstant > 0.0) {
            INDArray weightsView = weights.get(NDArrayIndex.interval(0,weights.rows()-1,false),NDArrayIndex.all());
            for(int c = 0; c < weights.columns(); c++) {
                // check norm columnwise (except for bias)
                double norm2 = weightsView.getColumn(c).norm2Number().doubleValue();
                if(norm2 > maxNormRegularizationConstant) {
                    // project onto ball
                    weightsView.getColumn(c).muli(maxNormRegularizationConstant/norm2);
                }
            }
        }
    }

    public void initLayer(NeuralNetwork network, boolean adaGrad, double learningRate, double momentum, WeightInit weightInit, ActivationFunction function) {
        this.network=network;
        if(!adaGradWasSet)this.adaGrad=adaGrad;
        if(!learningRateWasSet)this.learningRate=learningRate;
        if(!momentumWasSet)this.momentum=momentum;
        if(!weightInitWasSet)this.weightInit=weightInit;
        if(!activationWasSet)this.function=function;
    }

    public void setDelta(INDArray delta) {
        this.delta=delta;
    }

    public int getInputCount() {
        return inputCount;
    }

    public int getOutputCount() {
        return outputCount;
    }

    public INDArray getOutputs() {
        return outputs;
    }

    public INDArray getInputs() {
        return inputs;
    }

    public INDArray getNets() {
        return nets;
    }

    public void updateWeights() {
        assert delta!=null : "Please set weights delta before updating weights";
        regularizeWeights();

        INDArray inputsWithBias = MatrixHelper.addBias(this.inputs);
        INDArray deltaWeights = inputsWithBias.transpose().mmul(delta);
        if(network.getBatchSize()>0) deltaWeights.divi(inputsWithBias.rows());

        // adagrad handling
        if (adaGrad) {
            if (historicalDelta == null) {
                historicalDelta = Transforms.pow(deltaWeights, 2, true);
            } else {
                historicalDelta.addi(Transforms.pow(deltaWeights, 2, true));
            }
            deltaWeights.divi(Transforms.sqrt(historicalDelta, true).addi(MatrixHelper.FUDGE_FACTOR));
        }

        // learning rate factor
        deltaWeights.muli(learningRate);

        // momentum handling
        if (momentum > 0.0) {
            if (lastDeltaWeights != null) {
                deltaWeights.addi(lastDeltaWeights.mul(momentum));
            }
            lastDeltaWeights = deltaWeights;
        }

        // update weights
        this.weights.addi(deltaWeights);
    }

    public ActivationFunction getActivation() {
        return function;
    }

    // backpropagation
    public INDArray backProp(INDArray previousWeights, INDArray previousDelta, INDArray targetOutput) {
        delta = function.computeGradient(this, targetOutput, previousDelta, previousWeights);
        return delta;
    }

    // returns output matrix and net matrix
    public Pair<INDArray, INDArray> feedForward(INDArray inputs, boolean train) {
        assert weights.rows() == inputCount + 1 : "Weight.rows() must equal input count + 1";
        assert weights.columns() == outputCount : "Weight.columns() must equal output count";
        assert inputs.columns() == inputCount || negativeSampling > 0.0: "Input.columns must equal input count";
        // handle dropout
        this.inputs = inputs;

        INDArray weights = this.weights;
        if (dropConnect > 0.0) {
            if (train) {
                // multiply by bernoulli random variable
                int[] weightShape = weights.shape();
                INDArray dropOutVector = randomBernoulliVariables(weightShape, dropConnect);
                dropConnectWeights = weights.dup().muli(dropOutVector);
                weights = dropConnectWeights;
            }
        }
        INDArray inputsWithBias = MatrixHelper.addBias(this.inputs);

        // drop connect only if training
        nets = dropConnect > 0.0 && train ? getSampleNetOutputs(weights, inputsWithBias) : inputsWithBias.mmul(weights);
        outputs = function.activation(nets);
        return new Pair<>(outputs, nets);
    }

    public INDArray getSampleNetOutputs(INDArray weights, INDArray inputsWithBias) {
        // sample from normal distribution
        //assert dropConnectWeights != null : "Must be using dropConnect";
        assert numSamples > 0 : "Must have a positive number of samples";
        // expected pWv
        INDArray meanNets = inputsWithBias.mmul(weights);
        INDArray expectation = meanNets.mul(dropConnect);
        // variance p(1 − p)(W ⋆ W )(v ⋆ v)
        INDArray variance = Transforms.pow(inputsWithBias, 2, false).mmul(Transforms.pow(weights, 2, true)).muli(dropConnect * (1.0 - dropConnect));
        INDArray gaussians = Nd4j.randn(new int[]{inputsWithBias.rows(), outputCount, numSamples}, System.currentTimeMillis());
        INDArray samples = Nd4j.create(new int[]{inputsWithBias.rows(), outputCount, numSamples});
        for (int i = 0; i < inputsWithBias.rows(); i++) {
            for (int j = 0; j < outputCount; j++) {
                samples.put(new INDArrayIndex[]{NDArrayIndex.point(i),NDArrayIndex.point(j)}, gaussians.get(NDArrayIndex.point(i),NDArrayIndex.point(j)).addi(expectation.getDouble(i,j)).divi(Math.sqrt(variance.getDouble(i, j))));
            }
        }
        return samples.mean(2);
    }

    public INDArray getWeights() {
        return getWeights(false);
    }

    public INDArray getWeights(boolean train) {
        if (dropConnect > 0.0 && train) {
            return dropConnectWeights;
        }
        return weights;
    }

    public static class Builder {
        private FeedForwardLayer layer;
        public Builder(int inputCount, int outputCount) {
            this.layer=new FeedForwardLayer(inputCount,outputCount);
        }
        public Builder setDropConnect(double dropConnect) {
            assert dropConnect <= 1.0 : "dropConnect cannot be greater than 1";
            if (dropConnect == 1.0) dropConnect = -1.0; // efficiency
            layer.dropConnect=dropConnect;
            return this;
        }
        public Builder setMomentum(double momentum) {
            layer.momentum=momentum;
            layer.momentumWasSet=true;
            return this;
        }
        public Builder setActivation(ActivationFunction activation) {
            layer.activationWasSet=true;
            layer.function=activation;
            return this;
        }
        public Builder setWeightInit(WeightInit weightInit) {
            layer.weightInitWasSet=true;
            layer.weightInit=weightInit;
            return this;
        }
        public Builder setLearningRate(double learningRate) {
            layer.learningRate=learningRate;
            layer.learningRateWasSet=true;
            return this;
        }
        public Builder useAdaGrad(boolean adaGrad) {
            layer.adaGrad=adaGrad;
            layer.adaGradWasSet=true;
            return this;
        }
        public Builder setNumSamples(int samples) {
            layer.numSamples=samples;
            return this;
        }
        public Builder setMaxNormRegularization(double c) {
            layer.maxNormRegularizationConstant=c;
            return this;
        }
        public Layer build() {
            return layer;
        }
    }
}