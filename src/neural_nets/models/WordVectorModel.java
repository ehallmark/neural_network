package neural_nets.models;

import neural_nets.activations.ActivationFunction;
import neural_nets.activations.Sigmoid;
import neural_nets.activations.SoftMax;
import neural_nets.iterators.WordVectorDataSet;
import neural_nets.iterators.CSVTextIterator;
import neural_nets.iterators.TextIterator;
import neural_nets.layers.FeedForwardLayer;
import neural_nets.layers.Layer;
import neural_nets.listeners.Listener;
import neural_nets.listeners.WordVectorListener;
import neural_nets.stop_conditions.EarlyStopCondition;
import neural_nets.stop_conditions.StopOnClassError;
import neural_nets.utils.Vocabulary;
import neural_nets.weight_inits.WeightInit;
import neural_nets.weight_inits.XavierFanAvg;
import org.nd4j.linalg.api.ndarray.INDArray;

import java.io.*;
import java.sql.SQLException;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;

/**
 * Created by ehallmark on 11/13/16.
 */
public class WordVectorModel extends NeuralNetwork implements Serializable {
    public static File wordVectorModelFile = new File("word_vector_model.wv");
    private int vectorSize = 100;
    private double subSampling = -1.0;
    private int windowSize = 5;
    private double maxNormRegularization = -1.0;
    private Vocabulary vocab;
    private transient TextIterator trainIterator;
    private transient TextIterator testIterator;
    private transient TextIterator valIterator;
    private boolean resetVocab = true;
    private boolean predictParagraphVectors = false;

    private WordVectorModel() {
        super();
    }

    private void setupDataSet() {
        dataSet = new WordVectorDataSet.Builder(vocab,trainIterator,testIterator,valIterator,batchSize,predictParagraphVectors).setSubSampling(subSampling).resetVocabulary(resetVocab).setWindowSize(windowSize).build();
    }

    public INDArray getWordVector(String word) {
        int wordIdx = vocab.indexOfWord(word);
        if(wordIdx < 0) return null;
        return getOutputLayer().getWeights(false).getColumn(wordIdx);
    }

    public List<String> mostSimilarWordsTo(Vocabulary.Word word, int num) {
        INDArray weights = getOutputLayer().getWeights(false).dup();
        Map<Double,Integer> resultMap = new HashMap<>();
        INDArray resultArray = weights.transposei().mmul(weights.getColumn(word.getIndex()));
        double[] results = resultArray.data().asDouble();
        for(int i = 0; i < results.length-1; i++) {
            resultMap.put(results[i],i);
        }
        return resultMap.entrySet().stream().filter(e->!e.getValue().equals(word.getIndex())).sorted((e1,e2)->e2.getKey().compareTo(e1.getKey())).limit(num)
                .map(e->vocab.wordOrLabelFromIndex(e.getValue()))
                .filter(w->w!=null).map(w->w.getText()).collect(Collectors.toList());
    }

    public static class Builder extends NeuralNetwork.Builder {
        public Builder() {
            super(new WordVectorModel());
        }
        public Builder setTestIterator(TextIterator testIterator) {
            ((WordVectorModel)network).testIterator=testIterator;
            return this;
        }
        public Builder setValidationIterator(TextIterator validationIterator) {
            ((WordVectorModel)network).valIterator=validationIterator;
            return this;
        }
        public Builder setTrainingIterator(TextIterator trainingIterator) {
            ((WordVectorModel)network).trainIterator=trainingIterator;
            return this;
        }
        public Builder predictParagraphVectors(boolean predictParagraphVectors) {
            ((WordVectorModel)network).predictParagraphVectors=predictParagraphVectors;
            return this;
        }
        public Builder setVocabulary(Vocabulary vocab) {
            ((WordVectorModel)network).vocab=vocab;
            return this;
        }

        @Override
        public Builder numEpochs(int numEpochs) {
            super.numEpochs(numEpochs);
            return this;
        }

        @Override
        public Builder addLayer(Layer layer) {
            throw new UnsupportedOperationException("Cannot add custom layer to word vector model");
        }

        @Override
        public Builder plotGraphs(boolean actuallyPlotGraphs) {
            super.plotGraphs(actuallyPlotGraphs);
            return this;
        }

        @Override
        public Builder hugeModelExpected(boolean hugeModel) {
            super.hugeModelExpected(hugeModel);
            return this;
        }

        @Override
        public Builder useAdaGrad(boolean reallyUse) {
            super.useAdaGrad(reallyUse);
            return this;
        }

        @Override
        public Builder setWeightInit(WeightInit weightInit) {
            super.setWeightInit(weightInit);
            return this;
        }

        @Override
        public Builder setMomentum(double momentum) {
            super.setMomentum(momentum);
            return this;
        }

        @Override
        public Builder setListener(Listener listener) {
            super.setListener(listener);
            return this;
        }

        @Override
        public Builder setLearningRate(double learningRate) {
            super.setLearningRate(learningRate);
            return this;
        }

        @Override
        public Builder setStopCondition(EarlyStopCondition stopCondition) {
            super.setStopCondition(stopCondition);
            return this;
        }

        @Override
        public Builder setNumIterations(int num) {
            super.setNumIterations(num);
            return this;
        }

        @Override
        public Builder setActivation(ActivationFunction activation) {
            super.setActivation(activation);
            return this;
        }

        @Override
        public Builder setBatchSize(int size) {
            super.setBatchSize(size);
            return this;
        }

        public Builder setSubSampling(double t) {
            ((WordVectorModel)network).subSampling=t;
            return this;
        }
        public Builder setVectorSize(int vectorSize) {
            ((WordVectorModel)network).vectorSize=vectorSize;
            return this;
        }
        public Builder setWindowSize(int windowSize) {
            ((WordVectorModel)network).windowSize=windowSize;
            return this;
        }
        public Builder resetVocabulary(boolean reset) {
            ((WordVectorModel)network).resetVocab=reset;
            return this;
        }
        public Builder setMaxNormRegularization(double maxNorm) {
            ((WordVectorModel)network).maxNormRegularization=maxNorm;
            return this;
        }
        public WordVectorModel build() {
            assert ((WordVectorModel)network).vocab!=null;
            assert ((WordVectorModel)network).trainIterator!=null;

            ((WordVectorModel)network).setupDataSet();

            Vocabulary vocab = ((WordVectorModel)network).vocab;
            boolean predictParagraphVectors = ((WordVectorModel)network).predictParagraphVectors;

            int visibleLayerSize = vocab.vocabSize() + (predictParagraphVectors ? vocab.numDistinctLabels() : 0);
            int vectorSize = ((WordVectorModel)network).vectorSize;
            double maxNormRegularization = ((WordVectorModel)network).maxNormRegularization;

            network.layers.add(new FeedForwardLayer.Builder(visibleLayerSize,vectorSize).setActivation(new Sigmoid()).setMaxNormRegularization(maxNormRegularization).build());
            network.layers.add(new FeedForwardLayer.Builder(vectorSize,visibleLayerSize).setActivation(new SoftMax()).setMaxNormRegularization(maxNormRegularization).build());

            return (WordVectorModel)super.build();
        }
    }
}
