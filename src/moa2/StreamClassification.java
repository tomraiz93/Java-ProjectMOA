import java.util.ArrayList;
import java.util.Random;
import moa.classifiers.Classifier;
import moa.classifiers.trees.HoeffdingTree;
import moa.classifiers.bayes.NaiveBayes;
import weka.core.Attribute;
import weka.core.DenseInstance;
import weka.core.Instances;

public class StreamClassification {

    public static void main(String[] args) {
        // Tham số
        int numInstances = 200; // Số lượng instance
        int numAttributes = 3;  // Số lượng thuộc tính

        // Tạo danh sách thuộc tính
        ArrayList<Attribute> attributes = new ArrayList<>();
        for (int i = 0; i < numAttributes; i++) {
            attributes.add(new Attribute("attr" + (i + 1)));
        }
        
        // Danh sách lớp
        ArrayList<String> classValues = new ArrayList<>();
        classValues.add("Class1");
        classValues.add("Class2");
        attributes.add(new Attribute("class", classValues));

        // Tạo đối tượng Instances
        Instances data = new Instances("StreamData", attributes, 0);
        data.setClassIndex(data.numAttributes() - 1);
        Random rand = new Random();

        // Tạo dữ liệu giả lập
        for (int i = 0; i < numInstances; i++) {
            double[] values = new double[numAttributes + 1];
            for (int j = 0; j < numAttributes; j++) {
                values[j] = rand.nextDouble() * 10; // Giá trị thuộc tính ngẫu nhiên
            }
            // Gán lớp dựa trên điều kiện
            values[numAttributes] = (values[0] + values[1] > 10) ? 0 : 1; // Class1 nếu tổng lớn hơn 10
            data.add(new DenseInstance(1.0, values));
        }

        // Phân loại với Hoeffding Tree
        Classifier hoeffdingTree = new HoeffdingTree();
        hoeffdingTree.resetLearning();

        for (int i = 0; i < data.numInstances(); i++) {
            hoeffdingTree.trainOnInstance(data.instance(i));
        }

        // Tính độ chính xác cho Hoeffding Tree
        int correctHoeffding = 0;
        for (int i = 0; i < data.numInstances(); i++) {
            double[] votes = hoeffdingTree.getVotesForInstance(data.instance(i));
            double predictedClass = votes[0] > votes[1] ? 0 : 1;
            if (predictedClass == data.instance(i).classValue()) {
                correctHoeffding++;
            }
        }
        double accuracyHoeffding = (double) correctHoeffding / data.numInstances();
        System.out.println("Accuracy of Hoeffding Tree: " + accuracyHoeffding);

        // Phân loại với Naive Bayes
        Classifier naiveBayes = new NaiveBayes();
        naiveBayes.resetLearning();

        for (int i = 0; i < data.numInstances(); i++) {
            naiveBayes.trainOnInstance(data.instance(i));
        }

        // Tính độ chính xác cho Naive Bayes
        int correctNB = 0;
        for (int i = 0; i < data.numInstances(); i++) {
            double[] votes = naiveBayes.getVotesForInstance(data.instance(i));
            double predictedClass = votes[0] > votes[1] ? 0 : 1;
            if (predictedClass == data.instance(i).classValue()) {
                correctNB++;
            }
        }
        double accuracyNB = (double) correctNB / data.numInstances();
        System.out.println("Accuracy of Naive Bayes: " + accuracyNB);
    }
}