package moa1;
import java.util.ArrayList;
import java.util.Random;
import moa.classifiers.AbstractClassifier;
import moa.classifiers.Classifier;
import moa.classifiers.trees.HoeffdingTree;
import moa.classifiers.bayes.NaiveBayes;
import moa.classifiers.lazy.neighboursearch.LinearNNSearch;
import moa.core.Measurement;
import weka.core.Attribute;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.DenseInstance;
import weka.core.FastVector;

public class StreamClassification {

    public static class StreamKNN extends AbstractClassifier {

        private static final long serialVersionUID = 1L;

        private int k; // số lượng hàng xóm
        private int maxSize; // kích thước tối đa của cửa sổ
        private boolean useReservoir; // có sử dụng reservoir không

        private Instances window; // Cửa sổ các instance
        private int maxClassValue; // Giá trị class lớn nhất
        private int numProcessedInstances; // Số lượng instance đã xử lý

        protected long seed = 1;
        protected Random rand;

        public StreamKNN(int k, int maxSize, boolean useReservoir) {
            this.k = k;
            this.maxSize = maxSize;
            this.useReservoir = useReservoir;
            this.rand = new Random(seed);
        }

        @Override
        public void resetLearningImpl() {
            ArrayList<Attribute> attributes = new ArrayList<>();
            attributes.add(new Attribute("attr1"));
            attributes.add(new Attribute("attr2"));
            attributes.add(new Attribute("attr3"));
            FastVector classValues = new FastVector(2);
            classValues.addElement("Class1");
            classValues.addElement("Class2");
            attributes.add(new Attribute("class", classValues));

            window = new Instances("Window", attributes, 0);
            window.setClassIndex(window.numAttributes() - 1);

            maxClassValue = -1;
            numProcessedInstances = 0;
        }

        @Override
        public void trainOnInstanceImpl(Instance instance) {
            if (instance.classValue() > maxClassValue) {
                maxClassValue = (int) instance.classValue();
            }

            if (!useReservoir) {
                updateWindow(instance);
            } else {
                updateReservoir(instance);
            }
            numProcessedInstances++;
        }

        private void updateWindow(Instance instance) {
            if (window.numInstances() == maxSize) {
                window.remove(0);
            }
            window.add(instance);
        }

        private void updateReservoir(Instance instance) {
            if (window.numInstances() == maxSize) {
                int randomIndex = rand.nextInt(numProcessedInstances);
                if (randomIndex < maxSize) {
                    window.remove(randomIndex);
                    window.add(instance);
                }
            } else {
                window.add(instance);
            }
        }

        @Override
        public double[] getVotesForInstance(Instance instance) {
            LinearNNSearch search = new LinearNNSearch(window);
            Instances neighbours = null;

            try {
                neighbours = search.kNearestNeighbours(instance, Math.min(k, window.numInstances()));
            } catch (Exception e) {
                e.printStackTrace();
            }

            double votes[] = new double[maxClassValue + 1];
            for (int i = 0; i < neighbours.numInstances(); i++) {
                votes[(int) neighbours.instance(i).classValue()]++;
            }

            for (int i = 0; i < votes.length; i++) {
                votes[i] /= neighbours.numInstances();
            }

            return votes;
        }

        @Override
        public boolean isRandomizable() {
            return true;
        }

        @Override
        public void getModelDescription(StringBuilder sb, int indent) {
            sb.append("StreamKNN Classifier\n");
            sb.append("Number of neighbors (k): ").append(k).append("\n");
            sb.append("Max size of window: ").append(maxSize).append("\n");
            sb.append("Using reservoir sampling: ").append(useReservoir).append("\n");
        }

        @Override
        protected Measurement[] getModelMeasurementsImpl() {
            return new Measurement[0];
        }
    }

    public static void main(String[] args) {
        int k = 3; // số lượng hàng xóm
        int maxSize = 100; // kích thước tối đa của cửa sổ
        boolean useReservoir = false; // không sử dụng reservoir

        StreamKNN knn = new StreamKNN(k, maxSize, useReservoir);
        knn.resetLearningImpl();

        HoeffdingTree hoeffdingTree = new HoeffdingTree();
        hoeffdingTree.prepareForUse();

        NaiveBayes naiveBayes = new NaiveBayes();
        naiveBayes.prepareForUse();

        // Tạo danh sách thuộc tính
        ArrayList<Attribute> attributes = new ArrayList<>();
        attributes.add(new Attribute("attr1"));
        attributes.add(new Attribute("attr2"));
        attributes.add(new Attribute("attr3"));
        FastVector classValues = new FastVector(2);
        classValues.addElement("Class1");
        classValues.addElement("Class2");
        attributes.add(new Attribute("class", classValues));

        Instances dataset = new Instances("Dataset", attributes, 0);
        dataset.setClassIndex(dataset.numAttributes() - 1);

        // Tạo một số instance mẫu
        double[][] sampleData = {
            {1.0, 2.0, 3.0, 0}, // Class1
            {2.0, 3.0, 4.0, 0}, // Class1
            {3.0, 1.0, 2.0, 1}, // Class2
            {4.0, 5.0, 6.0, 1}  // Class2
        };

        for (double[] values : sampleData) {
            Instance instance = new DenseInstance(1.0, values);
            instance.setDataset(dataset); // Gán dataset cho instance
            knn.trainOnInstanceImpl(instance);
            hoeffdingTree.trainOnInstance(instance);
            naiveBayes.trainOnInstance(instance);
        }

        // Dự đoán cho một instance mới
        double[] newValues = {2.5, 3.5, 4.5, -1}; // Class chưa xác định
        Instance newInstance = new DenseInstance(1.0, newValues);
        newInstance.setDataset(dataset);

        double[] knnVotes = knn.getVotesForInstance(newInstance);
        double[] hoeffdingVotes = hoeffdingTree.getVotesForInstance(newInstance);
        double[] naiveBayesVotes = naiveBayes.getVotesForInstance(newInstance);

        System.out.println("KNN Votes:");
        for (double vote : knnVotes) {
            System.out.println(vote);
        }

        System.out.println("Hoeffding Tree Votes:");
        for (double vote : hoeffdingVotes) {
            System.out.println(vote);
        }

        System.out.println("Naive Bayes Votes:");
        for (double vote : naiveBayesVotes) {
            System.out.println(vote);
        }

        // Tính độ chính xác cho từng classifier
        int correctKNN = 0;
        int correctHoeffding = 0;
        int correctNB = 0;

        for (double[] values : sampleData) {
            Instance instance = new DenseInstance(1.0, values);
            instance.setDataset(dataset);

            double[] knnVote = knn.getVotesForInstance(instance);
            double knnPrediction = knnVote[0] > knnVote[1] ? 0 : 1;
            if (knnPrediction == instance.classValue()) {
                correctKNN++;
            }

            double[] hoeffdingVote = hoeffdingTree.getVotesForInstance(instance);
            double hoeffdingPrediction = hoeffdingVote[0] > hoeffdingVote[1] ? 0 : 1;
            if (hoeffdingPrediction == instance.classValue()) {
                correctHoeffding++;
            }

            double[] nbVote = naiveBayes.getVotesForInstance(instance);
            double nbPrediction = nbVote[0] > nbVote[1] ? 0 : 1;
            if (nbPrediction == instance.classValue()) {
                correctNB++;
            }
        }

        double accuracyKNN = (double) correctKNN / sampleData.length;
        double accuracyHoeffding = (double) correctHoeffding / sampleData.length;
        double accuracyNB = (double) correctNB / sampleData.length;

        System.out.println("Accuracy of StreamKNN: " + accuracyKNN);
        System.out.println("Accuracy of Hoeffding Tree: " + accuracyHoeffding);
        System.out.println("Accuracy of Naive Bayes: " + accuracyNB);
    }
}