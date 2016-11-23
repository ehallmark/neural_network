package neural_nets.iterators;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Iterator;
import java.util.List;

/**
 * Created by ehallmark on 11/15/16.
 */
public class LineTextIterator implements TextIterator {
    private Iterator<List<String>> iterator;
    private List<List<String>> underlyingData;
    private File file;

    public LineTextIterator(File file) {
        this.file=file;
        reset();
    }

    @Override
    public List<String> nextTokens() {
        return iterator.next();
    }

    @Override
    public List<String> currentLabels() {
        return null;
    }

    @Override
    public boolean hasNext() {
        return iterator.hasNext();
    }

    @Override
    public boolean reset() {
        if(underlyingData!=null) iterator = underlyingData.iterator();
        else {
            try (BufferedReader br = new BufferedReader(new FileReader(file))){
                Iterator<String> sentences = br.lines().iterator();
                underlyingData = new ArrayList<>();
                while(sentences.hasNext()) {
                    String text = sentences.next().toLowerCase().replaceAll("-", "").replaceAll("[^a-z ]", " ");
                    underlyingData.add(Arrays.asList(text.split("\\s+")));
                }
                reset();
            } catch (IOException e) {
                e.printStackTrace();
                throw new RuntimeException(e);
            }
        }
        return true;
    }
}
