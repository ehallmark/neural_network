package neural_nets.models;

import edu.stanford.nlp.util.Pair;
import neural_nets.layers.Layer;
import neural_nets.activations.ActivationFunction;
import neural_nets.activations.Sigmoid;
import neural_nets.iterators.Data;
import neural_nets.iterators.DataSet;
import neural_nets.iterators.DataIterator;
import neural_nets.listeners.Listener;
import neural_nets.stop_conditions.EarlyStopCondition;
import neural_nets.utils.Grapher;
import neural_nets.weight_inits.WeightInit;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.ops.transforms.Transforms;

import java.io.*;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.stream.Collectors;

/**
 * Created by ehallmark on 11/6/16.
 */
public class NeuralNetwork implements Serializable {
    protected transient DataSet dataSet;
    protected boolean plotGraphs = false;
    protected boolean hugeModel = false;
    protected boolean adaGrad = false;
    protected ActivationFunction activation = new Sigmoid();
    protected EarlyStopCondition stopCondition;
    protected double learningRate = 0.01;
    protected int currentEpoch;
    protected int numIterations = 1;
    protected INDArray actualClasses;
    protected double momentum = 0.0;
    protected WeightInit weightInit;
    protected Object listenerTarget = this;
    protected int numEpochs = 1000;
    protected List<Layer> layers;
    protected List<List<Double>> allClassErrors;
    protected int batchSize = 10;
    protected List<List<Double>> allErrors;
    protected transient Grapher grapher;
    protected transient Listener listener = null;


    protected NeuralNetwork(DataSet dataSet) {
        this.layers=new ArrayList<>();
        this.dataSet=dataSet;
    }

    // in case you don't have a dataset yet
    protected NeuralNetwork() {
        this.layers=new ArrayList();
    }

    public INDArray backPropagation(INDArray delta, INDArray targetOutput) {
        // update weights on output layer
        Layer outputLayer = layers.get(layers.size()-1);
        outputLayer.setDelta(delta);
        assert(outputLayer.getOutputs()!=null) : "Please train network before backpropagating!";
        INDArray previousWeights = outputLayer.getWeights(true);
        for(int i = layers.size()-2; i >= 0; i--) {
            Layer layer = layers.get(i);
            delta = layer.backProp(previousWeights,delta,targetOutput);
            previousWeights = layer.getWeights(true);
        }
        return delta;
    }

    // returns output matrix and net matrix
    public Pair<INDArray, INDArray> feedForward(INDArray inputs, boolean train) {
        Pair<INDArray, INDArray> data = new Pair<>(inputs,null);
        for(int i = 0; i < layers.size(); i++) {
            Layer layer = layers.get(i);
            data = layer.feedForward(data.first(), train);
        }
        return data;
    }

    // returns the error and classification error
    public Pair<Pair<Double,Double>,INDArray> evaluateNetworkError(Data inputData, boolean train) {
        // compute stuff
        actualClasses=inputData.getClasses();
        Pair<INDArray,INDArray> data = feedForward(inputData.getInputs(), train);
        INDArray outputs = data.first();
        INDArray classes = inputData.outputToClass(outputs);

        assert(classes.rows()==inputData.getClasses().rows()) : "classes and target classes have different row dimensions";
        assert(classes.columns()==inputData.getClasses().columns()) : "classes and target classes have different column dimensions";
        assert(outputs.rows()==inputData.getOutputs().rows()) : "outputs and target outputs have different row dimensions";
        assert(outputs.columns()==inputData.getOutputs().columns()) : "outputs and target outputs have different column dimensions";

        Layer outputLayer = layers.get(layers.size()-1);

        // get errors and error vectors
        INDArray targetOutputs = inputData.getOutputs();

        double error = Transforms.pow(outputs.sub(targetOutputs),2.0,false).sum(outputs.shape().length-1).meanNumber().doubleValue();
        double classError = new Double(Arrays.stream(classes.sub(inputData.getClasses()).data().asDouble()).filter(c->Math.abs(c)>=0.5d).count())/inputData.getClasses().rows();

        INDArray delta = outputLayer.getActivation().computeGradient(outputLayer,targetOutputs,null,null);
        return new Pair<>(new Pair<>(error,classError), delta);
    }

    // convenience
    public Pair<INDArray,DataSet> train() {
        return train(dataSet, numEpochs, numIterations);
    }

    // return weight matrix and update error values in DataSet class
    public Pair<INDArray,DataSet> train(DataSet dataSet, int numEpochs, int numIterations) {
        // set up errors
        List<Double> trainErrors = new ArrayList<>();
        List<Double> trainClassErrors = new ArrayList<>();
        List<Double> testErrors = new ArrayList<>();
        List<Double> testClassErrors = new ArrayList<>();
        List<Double> valErrors = new ArrayList<>();
        List<Double> valClassErrors = new ArrayList<>();
        this.allClassErrors= Arrays.asList(trainClassErrors,testClassErrors,valClassErrors);
        this.allErrors = Arrays.asList(trainErrors,testErrors,valErrors);

        // Loop for 0 <= count < numEpochs
        currentEpoch = 0;
        while(currentEpoch < numEpochs) {
            System.out.println("STARTING EPOCH ["+currentEpoch+"]");
            // training set
            iterateAndEvaluate(dataSet.getTrainingIterator(),trainErrors,trainClassErrors,currentEpoch,numIterations,true);
            if(!(trainClassErrors.size() > 0 && trainErrors.size()>0)) {
                throw new RuntimeException("Likely not enough data to get a full batch!");
            }

            // test set
            iterateAndEvaluate(dataSet.getTestIterator(),testErrors,testClassErrors,currentEpoch,numIterations,false);

            // validation set
            iterateAndEvaluate(dataSet.getValidationIterator(),valErrors,valClassErrors,currentEpoch,numIterations,false);

            // stopping condition
            if (stopCondition!=null) {
                if(stopCondition.stopEarly(this)) {
                    break;
                }
            }

            // Does the listener have anything to say?
            if(listener!=null)
                listener.printResults(this);


            currentEpoch++;
        }
        return new Pair<>(layers.get(layers.size()-1).getWeights(),dataSet);
    }

    private void iterateAndEvaluate(DataIterator iterator, List<Double> errors, List<Double> classErrors, int currentEpoch, int numIterations, boolean train) {
        if(iterator==null)return;
        iterator.reset();
        List<Double> batchErrors = new ArrayList<>();
        List<Double> batchClassErrors = new ArrayList<>();
        while(iterator.hasNext()) {
            Data datum = iterator.next();
            if (datum != null) {
                for(int i = 0; i < numIterations; i++) {
                    Pair<Pair<Double, Double>, INDArray> modelErrors = evaluateNetworkError(datum, train);
                    if (train) {
                        // get errors and error vectors
                        INDArray targetOutputs = datum.getOutputs();
                        backPropagation(modelErrors.second(), targetOutputs);
                        // update weights
                        for (Layer l : layers) {
                            l.updateWeights();
                        }
                    }
                    batchErrors.add(modelErrors.first().first());
                    batchClassErrors.add(modelErrors.first().second());

                    if(!hugeModel) {
                        datum.randomizeOrder();
                    }

                }
            }
        }
        if(!batchErrors.isEmpty()) {
            errors.add(batchErrors.stream().collect(Collectors.averagingDouble(d->d)));
        }
        if(!batchClassErrors.isEmpty()) {
            classErrors.add(batchClassErrors.stream().collect(Collectors.averagingDouble(d->d)));
        }
        // plot
        if (plotGraphs && grapher != null) grapher.updateData(this.allErrors, this.allClassErrors, currentEpoch);

    }

    public int getBatchSize() {
        return batchSize;
    }

    public List<List<Double>> getClassErrors() {
        return allClassErrors;
    }

    public List<List<Double>> getErrors() {
        return allErrors;
    }

    public int getNumIterations() {
        return numIterations;
    }

    public int getCurrentEpoch() {
        return currentEpoch;
    }

    public INDArray getActualClasses() {
        return actualClasses;
    }

    private void addLayer(Layer layer) {
        this.layers.add(layer);
    }

    public Layer getOutputLayer() {
        return layers.get(layers.size()-1);
    }

    public int indexOfLayer(Layer layer) {
        return layers.indexOf(layer);
    }

    public Layer getInputLayer() {
        return layers.get(0);
    }

    public Layer getPreviousLayer(Layer layer) {
        int idx = layers.indexOf(layer);
        if(idx > 0) return layers.get(idx-1);
        else return null;
    }

    public Layer getNextLayer(Layer layer) {
        int idx = layers.indexOf(layer);
        if(idx >= 0 && idx < layers.size()-1) return layers.get(idx+1);
        else return null;
    }

    // helper class to build neural network
    public static class Builder {
        protected NeuralNetwork network;
        public Builder(DataSet dataSet) {
            this.network = new NeuralNetwork(dataSet);
        }
        protected Builder(NeuralNetwork preBuilt) {
            this.network = preBuilt;
        }
        public Builder numEpochs(int numEpochs) {
            network.numEpochs=numEpochs;
            return this;
        }
        public Builder addLayer(Layer layer) {
            if(!network.layers.isEmpty()) {
                // check dimensions
                int previousOutputCount = network.layers.get(network.layers.size()-1).getOutputCount();
                if(previousOutputCount!=layer.getInputCount()) throw new RuntimeException("Invalid layer dimensions:\nPrevious Output: "+previousOutputCount+"\nInput: "+layer.getInputCount());
            }
            network.addLayer(layer);
            return this;
        }
        public Builder plotGraphs(boolean actuallyPlotGraphs) {
            network.plotGraphs=actuallyPlotGraphs;
            return this;
        }
        public Builder hugeModelExpected(boolean hugeModel) {
            network.hugeModel=hugeModel;
            return this;
        }
        public Builder useAdaGrad(boolean reallyUse) {
            network.adaGrad = reallyUse;
            return this;
        }
        public Builder setWeightInit(WeightInit weightInit) {
            network.weightInit=weightInit;
            return this;
        }
        public Builder setMomentum(double momentum) {
            network.momentum=momentum;
            return this;
        }
        public Builder setListener(Listener listener) {
            network.listener = listener;
            return this;
        }
        public Builder setLearningRate(double learningRate) {
            network.learningRate=learningRate;
            return this;
        }
        public Builder setStopCondition(EarlyStopCondition stopCondition) {
            network.stopCondition=stopCondition;
            return this;
        }
        public Builder setNumIterations(int num) {
            network.numIterations=num;
            return this;
        }
        public Builder setActivation(ActivationFunction activation) {
            network.activation = activation;
            return this;
        }
        public Builder setBatchSize(int size) {
            network.batchSize=size;
            return this;
        }
        public Builder setListenerTarget(Object target) {
            network.listenerTarget=target;
            return this;
        }
        public NeuralNetwork build() {
            if(network.layers.size()==0) throw new RuntimeException("Must have at least one layer!");
            if(network.layers.get(network.layers.size()-1).getOutputCount()!=network.dataSet.getOutputCount()) throw new RuntimeException("Output length is inconsistent in the last layer");
            // set Adaptive Gradients Option
            for(Layer l : network.layers) {
                // set layer defaults unless overwritten by layer
                l.initLayer(network, network.adaGrad, network.learningRate, network.momentum, network.weightInit, network.activation);
            }
            if(network.plotGraphs)network.grapher = new Grapher(network.numEpochs);
            return network;
        }
    }

}
