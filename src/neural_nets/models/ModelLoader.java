package neural_nets.models;

import java.io.*;

/**
 * Created by ehallmark on 11/17/16.
 */
public class ModelLoader<T> {
    public static void saveToFile(Object objToSave, File file) throws IOException {
        ObjectOutputStream oos = new ObjectOutputStream(new BufferedOutputStream(new FileOutputStream(file)));
        oos.writeObject(objToSave);
        oos.flush();
        oos.close();
    }

    public T loadFromFile(File file) throws IOException {
        T obj;
        try(ObjectInputStream ois = new ObjectInputStream(new BufferedInputStream(new FileInputStream(file)))) {
            obj = (T)ois.readObject();
        } catch(ClassNotFoundException cnfe) {
            cnfe.printStackTrace();
            return null;
        }
        return obj;
    }
}
