package neural_nets;

import neural_nets.activations.Sigmoid;
import neural_nets.activations.SoftMax;
import neural_nets.iterators.DataSet;
import neural_nets.iterators.CSVTextIterator;
import neural_nets.iterators.TextIterator;
import neural_nets.layers.FeedForwardLayer;
import neural_nets.listeners.WordVectorListener;
import neural_nets.models.NeuralNetwork;
import neural_nets.models.WordVectorModel;
import neural_nets.stop_conditions.StopOnClassError;
import neural_nets.models.ModelLoader;
import neural_nets.utils.MatrixHelper;
import neural_nets.utils.Vocabulary;
import neural_nets.weight_inits.XavierFanAvg;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.rng.DefaultRandom;
import org.nd4j.linalg.factory.Nd4j;

import java.io.File;
import java.sql.SQLException;
import java.util.Arrays;
import java.util.Collections;
import java.util.Random;

/**
 * Created by ehallmark on 11/6/16.
 */

public class TestNeuralNetworks {
    public static double to_d(boolean b) { return b ? 1.0 : 0.0; }
    public static double xOr(double[] array) { return to_d(!(Arrays.stream(array).allMatch(d->d>0.5)||Arrays.stream(array).noneMatch(d->d>0.5))); }
    public static double iff(double[] array) { return to_d((Arrays.stream(array).allMatch(d->d>0.5)||Arrays.stream(array).noneMatch(d->d>0.5))); }
    public static double Or(double[] array) { return to_d(!Arrays.stream(array).noneMatch(d->d>0.5)); }
    public static void runTests() {
        testOutputToClass();
        testAddBias();
        testBooleanLearning();
        /*try {
            testWordVectors();
        } catch(SQLException sql) {
            sql.printStackTrace();
        }*/
    }


    public static void testAddBias() {
        int rows = 20;
        int cols = 30;
        INDArray random = Nd4j.rand(rows,cols,-10,10,new DefaultRandom(System.currentTimeMillis()));
        INDArray withBias = MatrixHelper.addBias(random);
        assert withBias.rows()==rows :"Invalid rows in testAddBias";
        assert withBias.columns()==cols+1 : "Invalid cols in testAddBias";
        assert withBias.getColumn(cols).minNumber().doubleValue()==withBias.getColumn(cols).maxNumber().doubleValue() : "Last column should be all ones";
        assert withBias.getDouble(0,cols)==1.0: "Last column value should be a one";
    }


    public static void testWordVectors() throws SQLException {
        File testDataFile = new File("testdata.manual.2009.06.14.csv"); // SET
        File trainingDataFile = new File("training.1600000.processed.noemoticon.csv"); // SET
        trainingDataFile = testDataFile; // actual training set is too large
        int columnIdx = 5;
        TextIterator trainIter= new CSVTextIterator(trainingDataFile,columnIdx);
        TextIterator testIter= new CSVTextIterator(testDataFile,columnIdx);
        int batchSize = 100;
        CSVTextIterator valIter = null;
        Vocabulary vocab = new Vocabulary.Builder()
                .setMinWordFrequency(15)
                .setMaxDocumentFrequency(0.7)
                .setStopWords(Collections.emptySet())
                .build();
        boolean predictParagraphVectors = false;
        WordVectorModel model = new WordVectorModel.Builder()
                // Word Vector Options
                .resetVocabulary(true)
                .setSubSampling(-1)
                .setVectorSize(400)
                .setVocabulary(vocab)
                .setTrainingIterator(trainIter)
                .setTestIterator(testIter)
                .predictParagraphVectors(predictParagraphVectors)
                .setWindowSize(10)
                // Other Options
                .numEpochs(20)
                .plotGraphs(false)
                .useAdaGrad(true)
                //.setDropConnect(0.7)
                .hugeModelExpected(true)
                //.setMaxNormRegularization(1.0)
                .setStopCondition(new StopOnClassError(5))
                .setMomentum(0.8)
                .setBatchSize(batchSize)
                .setLearningRate(0.05)
                .setWeightInit(new XavierFanAvg())
                .setListener(new WordVectorListener(vocab,10))
                .build();
        model.train();

        try {
            ModelLoader.saveToFile(model,new File("my_word_vectors.wv"));
            model = null;
            model = new ModelLoader<WordVectorModel>().loadFromFile(new File("my_word_vectors.wv"));
            System.out.println("Word vector for data: "+model.getWordVector("data").toString());

        } catch(Exception e) {
            e.printStackTrace();
        }

    }

    public static void testBooleanLearning() {
        int numEpochs = 200;
        int numTests = 5;
        double momentum = 0.8;
        double learningRate = 0.05;

        int batchSize = 1;

        Random rand = new Random(System.currentTimeMillis());

        for(int i = 0; i < numTests; i++) {
            int inputCount = 2;
            int outputCount = 2;
            int trainingCount = 20;
            int testCount = 20;
            int maxNodes = 2+ Math.abs(rand.nextInt())%10;
            double[][] inputs = new double[trainingCount][];
            double[][] classes = new double[trainingCount][];
            double[][] testInputs = new double[testCount][];
            double[][] testClasses = new double[testCount][];
            for(int k = 0; k < trainingCount; k++) {
                inputs[k]=new double[inputCount];
                classes[k]=new double[1];
                for(int j = 0; j < inputCount; j++) {
                    inputs[k][j] = to_d(rand.nextBoolean());
                }
                classes[k][0] = xOr(inputs[k]);

                if(k < testCount) {
                    testInputs[k] = new double[inputCount];
                    for(int j = 0; j < inputCount; j++) {
                        testInputs[k][j] = to_d(rand.nextBoolean());
                    }
                    testClasses[k] = new double[]{xOr(testInputs[k])};

                }
            }
            DataSet dataSet = new DataSet.Builder(inputCount,outputCount,batchSize)
                    .setTrainingSet(Nd4j.create(inputs),Nd4j.create(classes),trainingCount)
                    .setTestSet(Nd4j.create(testInputs),Nd4j.create(testClasses),testCount)
                    .build();
            NeuralNetwork net = new NeuralNetwork.Builder(dataSet)
                    .plotGraphs(true)
                    .setLearningRate(learningRate)
                    .useAdaGrad(true)
                    .setStopCondition(new StopOnClassError(50))
                    .setMomentum(momentum)
                    .hugeModelExpected(false)
                    .setWeightInit(new XavierFanAvg())
                    .numEpochs(numEpochs)
                    .setBatchSize(batchSize)
                    .addLayer(new FeedForwardLayer.Builder(inputCount, 7).setActivation(new Sigmoid()).build())
                    //.addLayer(new Layer.Builder(7, 10).setMaxNormRegularization(2.0).setActivation(new Sigmoid()).setDropConnect(0.7).build())
                    //.addLayer(new Layer.Builder(10, 7).setActivation(new Sigmoid()).setDropConnect(0.7).build())
                    .addLayer(new FeedForwardLayer.Builder(7, 7).setActivation(new Sigmoid()).build())
                    //.addLayer(new FeedForwardLayer.Builder(20, 7).setActivation(new Sigmoid()).build())
                    //.addLayer(Math.abs(rand.nextInt()%maxNodes)  + 1, new Sigmoid())
                    .addLayer(new FeedForwardLayer.Builder(7,outputCount).setActivation(new Sigmoid()).build())
                    .build();

            net.train();

            try {
                ModelLoader.saveToFile(net,new File("my_test_network.nn"));
                net = null;
                net = new ModelLoader<NeuralNetwork>().loadFromFile(new File("my_test_network.nn"));

            } catch(Exception e) {
                e.printStackTrace();
            }


            System.out.println("Predicting XOR:");
            boolean train = false;
            System.out.println("[0, 0] => "+(net.feedForward(Nd4j.zeros(inputCount),train).first()));
            System.out.println("[0, 1] => "+(net.feedForward(Nd4j.create(new double[]{0.0,1.0}),train).first()));
            System.out.println("[1, 0] => "+(net.feedForward(Nd4j.create(new double[]{1.0,0.0}),train).first()));
            System.out.println("[1, 1] => "+(net.feedForward(Nd4j.ones(inputCount),train).first()));
        }

        System.out.println("Test passed");
    }

    public static void testOutputToClass() {

    }

    public static void main(String[] args) {
        runTests();
    }
}
