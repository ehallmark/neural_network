package neural_nets.iterators;

import com.opencsv.CSVIterator;
import com.opencsv.CSVReader;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.IOException;
import java.util.Arrays;
import java.util.List;
import java.util.concurrent.atomic.AtomicInteger;

/**
 * Created by ehallmark on 11/15/16.
 */
public class CSVTextIterator implements TextIterator{
    private File file;
    private CSVIterator iterator;
    private int colIdx;
    private AtomicInteger counter;

    public CSVTextIterator(File file, int colIdx) {
        this.file=file;
        this.colIdx = colIdx;
        this.counter = new AtomicInteger(0);
    }

    @Override
    public List<String> currentLabels() {
        return Arrays.asList(String.valueOf(counter.get()));
    }

    @Override
    public List<String> nextTokens() {
        counter.getAndIncrement();
        return Arrays.asList(iterator.next()[colIdx].toLowerCase().replaceAll("-","").replaceAll("[^a-z ]"," ").split("\\s+"));
    }

    @Override
    public boolean hasNext() {
        return iterator.hasNext();
    }

    @Override
    public boolean reset() {
        try {
            counter.set(0);
            this.iterator = new CSVIterator(new CSVReader(new BufferedReader(new FileReader(file))));
            return true;
        } catch(IOException ioe) {
            ioe.printStackTrace();
            throw new RuntimeException(ioe);
        }
    }
}
