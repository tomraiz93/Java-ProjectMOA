package moa1;

import javax.swing.*;
import java.awt.*;
import java.util.ArrayList;
import java.util.Random;
import moa.classifiers.trees.HoeffdingTree;
import moa.classifiers.bayes.NaiveBayes;
import weka.core.Attribute;
import weka.core.DenseInstance;
import weka.core.Instance;
import weka.core.Instances;
import org.jfree.chart.ChartFactory;
import org.jfree.chart.ChartPanel;
import org.jfree.chart.JFreeChart;
import org.jfree.chart.plot.PlotOrientation;
import org.jfree.data.category.DefaultCategoryDataset;
import moa.classifiers.AbstractClassifier;
import moa.classifiers.lazy.neighboursearch.LinearNNSearch;
import weka.core.FastVector;
import moa.core.Measurement;

public class StreamClassificationGUI {

    private StreamKNN knn;
    private HoeffdingTree hoeffdingTree;
    private NaiveBayes naiveBayes;
    private Instances dataset;
    private double[][] sampleData;
    private JPanel mainPanel;
    private JTextArea textArea;
    private ChartPanel chartPanel;
    private Random random;

    public static void main(String[] args) {
        SwingUtilities.invokeLater(() -> new StreamClassificationGUI().createAndShowGUI());
    }

    public StreamClassificationGUI() {
        int k = 3;
        int maxSize = 100;
        boolean useReservoir = false;

        knn = new StreamKNN(k, maxSize, useReservoir);
        knn.resetLearningImpl();

        hoeffdingTree = new HoeffdingTree();
        hoeffdingTree.prepareForUse();

        naiveBayes = new NaiveBayes();
        naiveBayes.prepareForUse();

        ArrayList<Attribute> attributes = new ArrayList<>();
        attributes.add(new Attribute("attr1"));
        attributes.add(new Attribute("attr2"));
        attributes.add(new Attribute("attr3"));
        FastVector classValues = new FastVector(2);
        classValues.addElement("Class1");
        classValues.addElement("Class2");
        attributes.add(new Attribute("class", classValues));

        dataset = new Instances("Dataset", attributes, 0);
        dataset.setClassIndex(dataset.numAttributes() - 1);

        random = new Random();
        generateSampleData(100);

        for (double[] values : sampleData) {
            Instance instance = new DenseInstance(1.0, values);
            instance.setDataset(dataset);
            knn.trainOnInstanceImpl(instance);
            hoeffdingTree.trainOnInstance(instance);
            naiveBayes.trainOnInstance(instance);
        }
    }

    private void generateSampleData(int numSamples) {
        sampleData = new double[numSamples][4];
        for (int i = 0; i < numSamples; i++) {
            double attr1 = random.nextDouble() * 10;
            double attr2 = random.nextDouble() * 10;
            double attr3 = random.nextDouble() * 10;
            double classValue = random.nextInt(2);
            sampleData[i][0] = attr1;
            sampleData[i][1] = attr2;
            sampleData[i][2] = attr3;
            sampleData[i][3] = classValue;
        }
    }

    private void createAndShowGUI() {
        JFrame frame = new JFrame("Stream Classification GUI");
        frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
        frame.setSize(800, 600);
        frame.setLayout(new BorderLayout());

        mainPanel = new JPanel(new BorderLayout());
        textArea = new JTextArea();
        textArea.setEditable(false);
        JScrollPane scrollPane = new JScrollPane(textArea);
        mainPanel.add(scrollPane, BorderLayout.CENTER);

        JPanel buttonPanel = new JPanel();
        JButton knnButton = new JButton("Run KNN");
        JButton hoeffdingButton = new JButton("Run Hoeffding Tree");
        JButton naiveBayesButton = new JButton("Run Naive Bayes");
        JButton compareButton = new JButton("Compare All");

        knnButton.addActionListener(e -> showAccuracy("KNN"));
        hoeffdingButton.addActionListener(e -> showAccuracy("Hoeffding Tree"));
        naiveBayesButton.addActionListener(e -> showAccuracy("Naive Bayes"));
        compareButton.addActionListener(e -> compareAll());

        buttonPanel.add(knnButton);
        buttonPanel.add(hoeffdingButton);
        buttonPanel.add(naiveBayesButton);
        buttonPanel.add(compareButton);

        frame.add(buttonPanel, BorderLayout.NORTH);
        frame.add(mainPanel, BorderLayout.CENTER);

        frame.setVisible(true);

        displayStreamData();
    }

    private void showAccuracy(String classifierName) {
        double accuracy = 0.0;
        switch (classifierName) {
            case "KNN":
                accuracy = calculateAccuracy(knn);
                break;
            case "Hoeffding Tree":
                accuracy = calculateAccuracy(hoeffdingTree);
                break;
            case "Naive Bayes":
                accuracy = calculateAccuracy(naiveBayes);
                break;
        }
        textArea.append(classifierName + " Accuracy: " + accuracy + "\n");
    }

    private void compareAll() {
        double accuracyKNN = calculateAccuracy(knn);
        double accuracyHoeffding = calculateAccuracy(hoeffdingTree);
        double accuracyNaiveBayes = calculateAccuracy(naiveBayes);

        DefaultCategoryDataset dataset = new DefaultCategoryDataset();
        dataset.addValue(accuracyKNN, "Accuracy", "KNN");
        dataset.addValue(accuracyHoeffding, "Accuracy", "Hoeffding Tree");
        dataset.addValue(accuracyNaiveBayes, "Accuracy", "Naive Bayes");

        JFreeChart barChart = ChartFactory.createBarChart(
                "Classifier Accuracy Comparison",
                "Classifier",
                "Accuracy",
                dataset,
                PlotOrientation.VERTICAL,
                true, true, false);

        if (chartPanel != null) {
            mainPanel.remove(chartPanel);
        }
        chartPanel = new ChartPanel(barChart);
        mainPanel.add(chartPanel, BorderLayout.SOUTH);
        mainPanel.revalidate();
        mainPanel.repaint();
    }

    private double calculateAccuracy(AbstractClassifier classifier) {
        int correct = 0;
        for (double[] values : sampleData) {
            Instance instance = new DenseInstance(1.0, values);
            instance.setDataset(dataset);

            double[] votes = classifier.getVotesForInstance(instance);
            double prediction = votes[0] > votes[1] ? 0 : 1;
            if (prediction == instance.classValue()) {
                correct++;
            }
        }
        return (double) correct / sampleData.length;
    }

    private void displayStreamData() {
        for (double[] values : sampleData) {
            textArea.append("Instance: ");
            for (double value : values) {
                textArea.append(value + " ");
            }
            textArea.append("\n");
        }
    }

    public static class StreamKNN extends AbstractClassifier {
        private static final long serialVersionUID = 1L;
        private int k;
        private int maxSize;
        private boolean useReservoir;
        private Instances window;
        private int maxClassValue;
        private int numProcessedInstances;
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
}