package neural_nets.listeners;

import neural_nets.models.NeuralNetwork;
import neural_nets.utils.Vocabulary;
import neural_nets.models.WordVectorModel;

import java.util.List;
import java.util.stream.Collectors;

/**
 * Created by ehallmark on 11/15/16.
 */
public class WordVectorListener implements Listener {
    private List<Vocabulary.Word> wordsToPredict;
    private Vocabulary vocab;
    private List<Double> trainClassErrors;
    private List<Double> testClassErrors;
    private List<Double> valClassErrors;
    private List<Double> trainErrors;
    private List<Double> testErrors;
    private List<Double> valErrors;
    private int numPredictions;

    public WordVectorListener(Vocabulary vocab, int numPredictions) {
        this.numPredictions=numPredictions;
        this.vocab=vocab;
    }

    @Override
    public void printResults(NeuralNetwork network) {
        WordVectorModel model = (WordVectorModel)network;
        assert vocab !=null : "Vocab is null";
        if(trainClassErrors==null)trainClassErrors=network.getClassErrors().get(0);
        if(testClassErrors==null)testClassErrors=network.getClassErrors().get(1);
        if(valClassErrors==null)valClassErrors=network.getClassErrors().get(2);
        if(trainErrors==null)trainErrors=network.getErrors().get(0);
        if(testErrors==null)testErrors=network.getErrors().get(1);
        if(valErrors==null)valErrors=network.getErrors().get(2);


        System.out.println("[EPOCH "+network.getCurrentEpoch()+"]");
        System.out.println();

        wordsToPredict=vocab.sampleWords(numPredictions);

        for(int i = 0; i < wordsToPredict.size(); i++) {
            Vocabulary.Word word = wordsToPredict.get(i);
            System.out.println("Most similar to "+word.getText()+": ["+String.join(",",model.mostSimilarWordsTo(word,numPredictions))+"]");
        }

        if(trainClassErrors.size()>=1) {
            System.out.println("Training error: " + String.join(", ",trainErrors.stream().map(e->String.valueOf(e)).collect(Collectors.toList())));
            System.out.println("Training class error: " + String.join(", ",trainClassErrors.stream().map(e->String.valueOf(e)).collect(Collectors.toList())));
        }
        if(testClassErrors.size()>=1) {
            System.out.println("Testing error: " + String.join(", ",testErrors.stream().map(e->String.valueOf(e)).collect(Collectors.toList())));
            System.out.println("Testing class error: " + String.join(", ",testClassErrors.stream().map(e->String.valueOf(e)).collect(Collectors.toList())));
        }
        if(valClassErrors.size()>=1) {
            System.out.println("Validation error: " + String.join(", ",valErrors.stream().map(e->String.valueOf(e)).collect(Collectors.toList())));
            System.out.println("Validation class error: " + String.join(", ",valClassErrors.stream().map(e->String.valueOf(e)).collect(Collectors.toList())));
        }

        System.out.println("Finished [EPOCH "+network.getCurrentEpoch()+"]");
        System.out.println();
    }
}
