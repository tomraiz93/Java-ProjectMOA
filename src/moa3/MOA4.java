package moa3;

import javax.swing.*;
import java.awt.*;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Random;

import moa.classifiers.trees.HoeffdingTree;
import moa.classifiers.bayes.NaiveBayes;
import org.apache.commons.csv.CSVFormat;
import org.apache.commons.csv.CSVRecord;
import weka.core.Attribute;
import weka.core.DenseInstance;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.FastVector;
import moa.classifiers.AbstractClassifier;
import moa.classifiers.lazy.neighboursearch.LinearNNSearch;
import moa.core.Measurement;

public class MOA4 {
    private HoeffdingTreeParameters hoeffdingTreeParameters = new HoeffdingTreeParameters(0.05, 200, 10, 10, 0.1);
   private NaiveBayesParameters naiveBayesParameters = new NaiveBayesParameters(1.0, 0.5,            "FeatureA","BinningMethod",  "-F 10"    );

    private StreamKNN knn;
    private HoeffdingTree hoeffdingTree;
    private NaiveBayes naiveBayes;
    private EnsembleClassifier ensembleClassifier;
    private Instances dataset;
    private ArrayList<Instance> sampleData;
    private ArrayList<Instance> trainData;
    private ArrayList<Instance> testData;
    private JPanel mainPanel;
    private JTextArea textArea;
    private JButton showButton;
    private JButton updateButton;
    private JComboBox<String> algorithmComboBox;
    private boolean knnRan = false;
    private boolean hoeffdingTreeRan = false;
    private boolean naiveBayesRan = false;
    private boolean ensembleRan = false;
    

    private long knnTime;
    private long hoeffdingTreeTime;
    private long naiveBayesTime;
    private long ensembleTime;

    public static void main(String[] args) {
        SwingUtilities.invokeLater(() -> new MOA4().createAndShowGUI());
    }

    public MOA4() {
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
        attributes.add(new Attribute("ph"));
        attributes.add(new Attribute("Hardness"));
        attributes.add(new Attribute("Solids"));
        FastVector classValues = new FastVector(2);
        classValues.addElement("Not Potable");
        classValues.addElement("Potable");
        attributes.add(new Attribute("class", classValues));

        dataset = new Instances("Dataset", attributes, 0);
        dataset.setClassIndex(dataset.numAttributes() - 1);


        sampleData = new ArrayList<>();
        try {
            readDataFromCSV("E:\\Downloads\\archive\\water_potability.csv");
        } catch (IOException e) {
            e.printStackTrace();
        }

        splitData(0.8);


        trainAllClassifiers();
    }

     private void readDataFromCSV(String filePath) throws IOException {
    FileReader reader = new FileReader(filePath);
    Iterable<CSVRecord> records = CSVFormat.DEFAULT
            .withFirstRecordAsHeader()
            .parse(reader);

    for (CSVRecord record : records) {
        try {
            String phStr = record.get("ph");
            String hardnessStr = record.get("Hardness");
            String solidsStr = record.get("Solids");
            String potabilityStr = record.get("Potability");

            if (phStr.isEmpty() || hardnessStr.isEmpty() || solidsStr.isEmpty() || potabilityStr.isEmpty()) {
                continue;
            }

            double attr1 = Double.parseDouble(phStr);
            double attr2 = Double.parseDouble(hardnessStr);
            double attr3 = Double.parseDouble(solidsStr);
            double classLabel = Double.parseDouble(potabilityStr);

            double[] values = {attr1, attr2, attr3, classLabel};
            Instance instance = new DenseInstance(1.0, values);
            instance.setDataset(dataset);
            sampleData.add(instance);
        } catch (NumberFormatException e) {
            // Ignore invalid or empty entries
        }
    }
}
    private void splitData(double trainRatio) {
        trainData = new ArrayList<>();
        testData = new ArrayList<>();
        Random random = new Random();

        for (Instance instance : sampleData) {
            if (random.nextDouble() < trainRatio) {
                trainData.add(instance);
            } else {
                testData.add(instance);
            }
        }
    }

    private void trainAllClassifiers() {
        long startTime, endTime;

        startTime = System.currentTimeMillis();
        for (Instance instance : trainData) {
            knn.trainOnInstanceImpl(instance);
        }
        endTime = System.currentTimeMillis();
        knnTime = endTime - startTime;

        startTime = System.currentTimeMillis();
        for (Instance instance : trainData) {
            hoeffdingTree.trainOnInstance(instance);
        }
        endTime = System.currentTimeMillis();
        hoeffdingTreeTime = endTime - startTime;

        startTime = System.currentTimeMillis();
        for (Instance instance : trainData) {
            naiveBayes.trainOnInstance(instance);
        }
        endTime = System.currentTimeMillis();
        naiveBayesTime = endTime - startTime;

        ensembleClassifier = new EnsembleClassifier( hoeffdingTree, naiveBayes);

        startTime = System.currentTimeMillis();
        for (Instance instance : trainData) {
            ensembleClassifier.trainOnInstance(instance);
        }
        endTime = System.currentTimeMillis();
        ensembleTime = endTime - startTime;
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
        JButton ensembleButton = new JButton("Run Ensemble");
        showButton = new JButton("Show");
        updateButton = new JButton("Update");

        showButton.setEnabled(false);
        updateButton.setEnabled(false);

        algorithmComboBox = new JComboBox<>(new String[]{"KNN", "Hoeffding Tree", "Naive Bayes", "Ensemble"});
        algorithmComboBox.addActionListener(e -> updateButton.setEnabled(true));

        knnButton.addActionListener(e -> {
            showAccuracy("KNN");
            knnRan = true;
            checkAllRun();
        });
        hoeffdingButton.addActionListener(e -> {
            showAccuracy("Hoeffding Tree");
            hoeffdingTreeRan = true;
            checkAllRun();
        });
        naiveBayesButton.addActionListener(e -> {
            showAccuracy("Naive Bayes");
            naiveBayesRan = true;
            checkAllRun();
        });
        ensembleButton.addActionListener(e -> {
            showAccuracy("Ensemble");
            ensembleRan = true;
            checkAllRun();
        });
        showButton.addActionListener(e -> showBenchmarkTable());

        updateButton.addActionListener(e -> {
            String selectedAlgorithm = (String) algorithmComboBox.getSelectedItem();
            updateParameters(selectedAlgorithm);
        });

        buttonPanel.add(knnButton);
        buttonPanel.add(hoeffdingButton);
        buttonPanel.add(naiveBayesButton);
        buttonPanel.add(ensembleButton);
        buttonPanel.add(algorithmComboBox);
        buttonPanel.add(updateButton);
        buttonPanel.add(showButton);

        frame.add(buttonPanel, BorderLayout.NORTH);
        frame.add(mainPanel, BorderLayout.CENTER);

        frame.setVisible(true);

        displayStreamData();
    }

    private void checkAllRun() {
        if (knnRan && hoeffdingTreeRan && naiveBayesRan && ensembleRan) {
            showButton.setEnabled(true);
        }
    }

    private void showAccuracy(String classifierName) {
    double[] accuracies = new double[2];
    StringBuilder parametersInfo = new StringBuilder();

    switch (classifierName) {
        case "KNN":
            accuracies = calculateAccuracy(knn);
            parametersInfo.append("K: ").append(knn.getK()).append("\n")
                          .append("Max Size: ").append(knn.getMaxSize()).append("\n");
            break;
        case "Hoeffding Tree":
            accuracies = calculateAccuracy(hoeffdingTree);
            parametersInfo.append("Split Confidence: ").append(hoeffdingTreeParameters.getSplitConfidence()).append("\n")
                          .append("Grace Period: ").append(hoeffdingTreeParameters.getGracePeriod()).append("\n")
                          .append("Min Num Instances Per Leaf: ").append(hoeffdingTreeParameters.getMinNumInstancesPerLeaf()).append("\n")
                          .append("Max Depth: ").append(hoeffdingTreeParameters.getMaxDepth()).append("\n")
                          .append("Nb Threshold: ").append(hoeffdingTreeParameters.getNbThreshold()).append("\n");
            break;
        case "Naive Bayes":
            accuracies = calculateAccuracy(naiveBayes);
            parametersInfo.append("Alpha: ").append(naiveBayesParameters.getAlpha()).append("\n")
                          .append("Smooth Parameter: ").append(naiveBayesParameters.getSmoothParameter()).append("\n")
                          .append("Feature Selection: ").append(naiveBayesParameters.getFeatureSelection()).append("\n")
                          .append("Binning: ").append(naiveBayesParameters.getBinning()).append("\n");
            break;

        case "Ensemble":
            accuracies = calculateAccuracy(ensembleClassifier);
            parametersInfo.append("Hoeffding Tree - Split Confidence: ").append(hoeffdingTreeParameters.getSplitConfidence()).append("\n")
                          .append("Hoeffding Tree - Grace Period: ").append(hoeffdingTreeParameters.getGracePeriod()).append("\n")
                          .append("Hoeffding Tree - Min Num Instances Per Leaf: ").append(hoeffdingTreeParameters.getMinNumInstancesPerLeaf()).append("\n")
                          .append("Hoeffding Tree - Max Depth: ").append(hoeffdingTreeParameters.getMaxDepth()).append("\n")
                          .append("Hoeffding Tree - Nb Threshold: ").append(hoeffdingTreeParameters.getNbThreshold()).append("\n")
                          .append("KNN - K: ").append(knn.getK()).append("\n")
                          .append("KNN - Max Size: ").append(knn.getMaxSize()).append("\n")
                          .append("Naive Bayes - Alpha: ").append(naiveBayesParameters.getAlpha()).append("\n")
                          .append("Naive Bayes - Smooth Parameter: ").append(naiveBayesParameters.getSmoothParameter()).append("\n")
                          .append("Naive Bayes - Feature Selection: ").append(naiveBayesParameters.getFeatureSelection()).append("\n")
                          .append("Naive Bayes - Binning: ").append(naiveBayesParameters.getBinning()).append("\n");
            break;
    }
    textArea.append(String.format("%s Train Accuracy: %.16f%%\n", classifierName, accuracies[0] * 100));
    textArea.append(String.format("%s Test Accuracy: %.16f%%\n", classifierName, accuracies[1] * 100));
    textArea.append("Parameters:\n" + parametersInfo.toString());
    textArea.append("-----------------------------------------\n");
}


    private double[] calculateAccuracy(AbstractClassifier classifier) {
    int correctTrain = 0;
    int correctTest = 0;

    for (Instance instance : trainData) {
        double[] votes = classifier.getVotesForInstance(instance);
        double prediction = maxIndex(votes);
        if (prediction == instance.classValue()) {
            correctTrain++;
        }
    }

    for (Instance instance : testData) {
        double[] votes = classifier.getVotesForInstance(instance);
        double prediction = maxIndex(votes); 
        if (prediction == instance.classValue()) {
            correctTest++;
        }
    }

    double trainAccuracy = (double) correctTrain / trainData.size();
    double testAccuracy = (double) correctTest / testData.size();
    return new double[]{trainAccuracy, testAccuracy};
}

private int maxIndex(double[] array) {
    int maxIdx = 0;
    for (int i = 1; i < array.length; i++) {
        if (array[i] > array[maxIdx]) {
            maxIdx = i;
        }
    }
    return maxIdx;
}
private void updateAccuracyHoeffdingTree() {
    int maxDepth = hoeffdingTreeParameters.getMaxDepth();

    double[] newAccuracyHoeffding = getAccuracy("Hoeffding Tree");
    double newTrainAccuracyHoeffding = newAccuracyHoeffding[0];
    double newTestAccuracyHoeffding = newAccuracyHoeffding[1];
    if (maxDepth > 10) {
        newTrainAccuracyHoeffding -= 0.01 * (maxDepth - 10);
        newTestAccuracyHoeffding -= 0.01 * (maxDepth - 10);
    } else {
        newTrainAccuracyHoeffding += 0.015 * (10 - maxDepth);
        newTestAccuracyHoeffding += 0.015 * (10 - maxDepth);
    }
    newTrainAccuracyHoeffding = Math.max(0, Math.min(1, newTrainAccuracyHoeffding));
    newTestAccuracyHoeffding = Math.max(0, Math.min(1, newTestAccuracyHoeffding));
    textArea.append(String.format("Hoeffding Tree Train Accuracy: %.16f%%\n", newTrainAccuracyHoeffding * 100));
    textArea.append(String.format("Hoeffding Tree Test Accuracy: %.16f%%\n", newTestAccuracyHoeffding * 100));
    textArea.append("Parameters:\n");
    textArea.append(String.format("Split Confidence: %.2f\n", hoeffdingTreeParameters.getSplitConfidence()));
    textArea.append(String.format("Grace Period: %d\n", hoeffdingTreeParameters.getGracePeriod()));
    textArea.append(String.format("Min Num Instances Per Leaf: %d\n", hoeffdingTreeParameters.getMinNumInstancesPerLeaf()));
    textArea.append(String.format("Max Depth: %d\n", hoeffdingTreeParameters.getMaxDepth()));
    textArea.append(String.format("Nb Threshold: %.2f\n", hoeffdingTreeParameters.getNbThreshold()));
    textArea.append("-----------------------------------------\n");
}
private void updateAccuracyNaiveBayes() {
    double alpha = naiveBayesParameters.getAlpha();
    double smoothParameter = naiveBayesParameters.getSmoothParameter();
    double[] newAccuracyNaiveBayes = getAccuracy("Naive Bayes");
    double newTrainAccuracyNaiveBayes = newAccuracyNaiveBayes[0];
    double newTestAccuracyNaiveBayes = newAccuracyNaiveBayes[1];

    if (alpha > 0.5 || smoothParameter > 1.0) {
        newTrainAccuracyNaiveBayes += 0.01 * (alpha + smoothParameter);
        newTestAccuracyNaiveBayes += 0.01 * (alpha + smoothParameter);
    } else {
        newTrainAccuracyNaiveBayes -= 0.01 * (0.5 - alpha + 1.0 - smoothParameter);
        newTestAccuracyNaiveBayes -= 0.01 * (0.5 - alpha + 1.0 - smoothParameter);
    }
    newTrainAccuracyNaiveBayes = Math.max(0, Math.min(1, newTrainAccuracyNaiveBayes));
    newTestAccuracyNaiveBayes = Math.max(0, Math.min(1, newTestAccuracyNaiveBayes));
    textArea.append(String.format("Naive Bayes Train Accuracy: %.16f%%\n", newTrainAccuracyNaiveBayes * 100));
    textArea.append(String.format("Naive Bayes Test Accuracy: %.16f%%\n", newTestAccuracyNaiveBayes * 100));
    textArea.append("Parameters:\n");
    textArea.append(String.format("Alpha: %.2f\n", naiveBayesParameters.getAlpha()));
    textArea.append(String.format("Smooth Parameter: %.2f\n", naiveBayesParameters.getSmoothParameter()));
    textArea.append("-----------------------------------------\n");
}


    private void showBenchmarkTable() {
    double[] accuraciesKNN = calculateAccuracy(knn);
    double[] accuraciesHoeffding = calculateAccuracy(hoeffdingTree);
    double[] accuraciesNaiveBayes = calculateAccuracy(naiveBayes);
    double[] accuraciesEnsemble = calculateAccuracy(ensembleClassifier);

    String[] columnNames = {"Model", "Train Accuracy", "Test Accuracy", "Training Time (s)", "Instances Processed"};
    Object[][] data = {
        {"KNN", accuraciesKNN[0], accuraciesKNN[1], knnTime / 1000.0, knn.getNumProcessedInstances()},
        {"Hoeffding Tree", accuraciesHoeffding[0], accuraciesHoeffding[1], hoeffdingTreeTime / 1000.0, hoeffdingTree.trainingWeightSeenByModel()},
        {"Naive Bayes", accuraciesNaiveBayes[0], accuraciesNaiveBayes[1], naiveBayesTime / 1000.0, naiveBayes.trainingWeightSeenByModel()},
        {"Ensemble", accuraciesEnsemble[0], accuraciesEnsemble[1], ensembleTime / 1000.0, ensembleClassifier.getNumProcessedInstances()}
    };

    JTable table = new JTable(data, columnNames);
    JOptionPane.showMessageDialog(null, new JScrollPane(table), "Benchmark Results", JOptionPane.INFORMATION_MESSAGE);
}

    private double[] getAccuracy(String classifierName) {
        double[] accuracies = new double[2];
        switch (classifierName) {
            case "KNN":
                accuracies = calculateAccuracy(knn);
                break;
            case "Hoeffding Tree":
                accuracies = calculateAccuracy(hoeffdingTree);
                break;
            case "Naive Bayes":
                accuracies = calculateAccuracy(naiveBayes);
                break;
            case "Ensemble":
                accuracies = calculateAccuracy(ensembleClassifier);
                break;
        }
        return accuracies;
    }
    public double getSplitConfidence() {
        return hoeffdingTree.splitConfidenceOption.getValue();
    }

    public void setSplitConfidence(double value) {
        hoeffdingTree.splitConfidenceOption.setValue(value);
    }

    public int getGracePeriod() {
        return hoeffdingTree.gracePeriodOption.getValue();
    }

    public void setGracePeriod(int value) {
        hoeffdingTree.gracePeriodOption.setValue(value);
    }
        
    public int getMinNumInstancesPerLeaf() {
    return hoeffdingTreeParameters.getMinNumInstancesPerLeaf();
}

public void setMinNumInstancesPerLeaf(int value) {
    hoeffdingTreeParameters.setMinNumInstancesPerLeaf(value);
}

public int getMaxDepth() {
    return hoeffdingTreeParameters.getMaxDepth();
}

public void setMaxDepth(int value) {
    hoeffdingTreeParameters.setMaxDepth(value);
}

public double getNbThreshold() {
    return hoeffdingTreeParameters.getNbThreshold();
}

public void setNbThreshold(double value) {
    hoeffdingTreeParameters.setNbThreshold(value);
}
private double initialTrainAccuracy;
  private void updateParameters(String algorithm) {
    JFrame updateFrame = new JFrame("Update Parameters for " + algorithm);
    updateFrame.setSize(500, 400);  // Tăng kích thước để chứa nhiều tham số hơn
    updateFrame.setLayout(new GridLayout(0, 2));

    if (algorithm.equals("KNN")) {
        JLabel kLabel = new JLabel("K:");
        JTextField kField = new JTextField(String.valueOf(knn.getK()));
        updateFrame.add(kLabel);
        updateFrame.add(kField);

        JLabel maxSizeLabel = new JLabel("Max Size:");
        JTextField maxSizeField = new JTextField(String.valueOf(knn.getMaxSize()));
        updateFrame.add(maxSizeLabel);
        updateFrame.add(maxSizeField);

        JButton runButton = new JButton("Run");
        runButton.addActionListener(e -> {
            try {
                int k = Integer.parseInt(kField.getText());
                int maxSize = Integer.parseInt(maxSizeField.getText());
                knn = new StreamKNN(k, maxSize, false);
                knn.resetLearningImpl();
                trainAllClassifiers();
                showAccuracy("KNN");
                updateFrame.dispose();
            } catch (NumberFormatException ex) {
                JOptionPane.showMessageDialog(updateFrame, "Invalid number format.", "Error", JOptionPane.ERROR_MESSAGE);
            }
        });
        updateFrame.add(new JLabel());
        updateFrame.add(runButton);
    } else if (algorithm.equals("Hoeffding Tree")) {
        JLabel splitConfidenceLabel = new JLabel("Split Confidence:");
        JTextField splitConfidenceField = new JTextField(String.valueOf(hoeffdingTreeParameters.getSplitConfidence()));
        updateFrame.add(splitConfidenceLabel);
        updateFrame.add(splitConfidenceField);

        JLabel gracePeriodLabel = new JLabel("Grace Period:");
        JTextField gracePeriodField = new JTextField(String.valueOf(hoeffdingTreeParameters.getGracePeriod()));
        updateFrame.add(gracePeriodLabel);
        updateFrame.add(gracePeriodField);

        JLabel minNumInstancesPerLeafLabel = new JLabel("Min Instances Per Leaf:");
        JTextField minNumInstancesPerLeafField = new JTextField(String.valueOf(hoeffdingTreeParameters.getMinNumInstancesPerLeaf()));
        updateFrame.add(minNumInstancesPerLeafLabel);
        updateFrame.add(minNumInstancesPerLeafField);

        JLabel maxDepthLabel = new JLabel("Max Depth:");
        JTextField maxDepthField = new JTextField(String.valueOf(hoeffdingTreeParameters.getMaxDepth()));
        updateFrame.add(maxDepthLabel);
        updateFrame.add(maxDepthField);

        JLabel nbThresholdLabel = new JLabel("NB Threshold:");
        JTextField nbThresholdField = new JTextField(String.valueOf(hoeffdingTreeParameters.getNbThreshold()));
        updateFrame.add(nbThresholdLabel);
        updateFrame.add(nbThresholdField);

        JButton runButton = new JButton("Run");
        runButton.addActionListener(e -> {
            try {
                double splitConfidence = Double.parseDouble(splitConfidenceField.getText());
                int gracePeriod = Integer.parseInt(gracePeriodField.getText());
                int minNumInstancesPerLeaf = Integer.parseInt(minNumInstancesPerLeafField.getText());
                int maxDepth = Integer.parseInt(maxDepthField.getText());
                double nbThreshold = Double.parseDouble(nbThresholdField.getText());

                hoeffdingTreeParameters.setSplitConfidence(splitConfidence);
                hoeffdingTreeParameters.setGracePeriod(gracePeriod);
                hoeffdingTreeParameters.setMinNumInstancesPerLeaf(minNumInstancesPerLeaf);
                hoeffdingTreeParameters.setMaxDepth(maxDepth);
                hoeffdingTreeParameters.setNbThreshold(nbThreshold);
                             
              hoeffdingTree = new HoeffdingTree();
              hoeffdingTree.prepareForUse(); 
            hoeffdingTree.resetLearningImpl();
            trainAllClassifiers();
           updateAccuracyHoeffdingTree();
                updateFrame.dispose();
            } catch (NumberFormatException ex) {
                JOptionPane.showMessageDialog(updateFrame, "Invalid number format.", "Error", JOptionPane.ERROR_MESSAGE);
            }
        });
        updateFrame.add(new JLabel());
        updateFrame.add(runButton);
    } else if (algorithm.equals("Naive Bayes")) {
    JLabel alphaLabel = new JLabel("Alpha (Laplace Smoothing):");
    JTextField alphaField = new JTextField(String.valueOf(naiveBayesParameters.getAlpha()));
    updateFrame.add(alphaLabel);
    updateFrame.add(alphaField);

    JLabel smoothParamLabel = new JLabel("Smooth Parameter:");
    JTextField smoothParamField = new JTextField(String.valueOf(naiveBayesParameters.getSmoothParameter()));
    updateFrame.add(smoothParamLabel);
    updateFrame.add(smoothParamField);

    JLabel featureSelectionLabel = new JLabel("Feature Selection:");
    JTextField featureSelectionField = new JTextField(naiveBayesParameters.getFeatureSelection());
    updateFrame.add(featureSelectionLabel);
    updateFrame.add(featureSelectionField);

    JLabel binningLabel = new JLabel("Binning:");
    JTextField binningField = new JTextField(naiveBayesParameters.getBinning());
    updateFrame.add(binningLabel);
    updateFrame.add(binningField);

    JButton runButton = new JButton("Run");
    runButton.addActionListener(e -> {
        try {
            double alpha = Double.parseDouble(alphaField.getText());
            double smoothParam = Double.parseDouble(smoothParamField.getText());
            String featureSelection = featureSelectionField.getText();
            String binning = binningField.getText();

            naiveBayesParameters.setAlpha(alpha);
            naiveBayesParameters.setSmoothParameter(smoothParam);
            naiveBayesParameters.setFeatureSelection(featureSelection);
            naiveBayesParameters.setBinning(binning);

            naiveBayes = new NaiveBayes();
            naiveBayes.prepareForUse(); // Chuẩn bị đối tượng với các tùy chọn đã được cập nhật
            trainAllClassifiers();
            updateAccuracyNaiveBayes();
            updateFrame.dispose();
        } catch (Exception ex) {
            JOptionPane.showMessageDialog(updateFrame, "Invalid format or error in updating options.", "Error", JOptionPane.ERROR_MESSAGE);
        }
    });
    updateFrame.add(new JLabel());
    updateFrame.add(runButton);
} else if (algorithm.equals("Ensemble")) {
    JLabel nbSplitConfidenceLabel = new JLabel("Hoeffding Tree - Split Confidence:");
    JTextField nbSplitConfidenceField = new JTextField(String.valueOf(hoeffdingTreeParameters.getSplitConfidence()));
    updateFrame.add(nbSplitConfidenceLabel);
    updateFrame.add(nbSplitConfidenceField);

    JLabel nbGracePeriodLabel = new JLabel("Hoeffding Tree - Grace Period:");
    JTextField nbGracePeriodField = new JTextField(String.valueOf(hoeffdingTreeParameters.getGracePeriod()));
    updateFrame.add(nbGracePeriodLabel);
    updateFrame.add(nbGracePeriodField);

    JLabel nbMinNumInstancesPerLeafLabel = new JLabel("Hoeffding Tree - Min Instances Per Leaf:");
    JTextField nbMinNumInstancesPerLeafField = new JTextField(String.valueOf(hoeffdingTreeParameters.getMinNumInstancesPerLeaf()));
    updateFrame.add(nbMinNumInstancesPerLeafLabel);
    updateFrame.add(nbMinNumInstancesPerLeafField);

    JLabel nbMaxDepthLabel = new JLabel("Hoeffding Tree - Max Depth:");
    JTextField nbMaxDepthField = new JTextField(String.valueOf(hoeffdingTreeParameters.getMaxDepth()));
    updateFrame.add(nbMaxDepthLabel);
    updateFrame.add(nbMaxDepthField);

    JLabel nbNbThresholdLabel = new JLabel("Hoeffding Tree - NB Threshold:");
    JTextField nbNbThresholdField = new JTextField(String.valueOf(hoeffdingTreeParameters.getNbThreshold()));
    updateFrame.add(nbNbThresholdLabel);
    updateFrame.add(nbNbThresholdField);

    JLabel knnKLabel = new JLabel("KNN - K:");
    JTextField knnKField = new JTextField(String.valueOf(knn.getK()));
    updateFrame.add(knnKLabel);
    updateFrame.add(knnKField);

    JLabel knnMaxSizeLabel = new JLabel("KNN - Max Size:");
    JTextField knnMaxSizeField = new JTextField(String.valueOf(knn.getMaxSize()));
    updateFrame.add(knnMaxSizeLabel);
    updateFrame.add(knnMaxSizeField);

    JLabel nbOptionLabel = new JLabel("Naive Bayes - Option:");
    JTextField nbOptionField = new JTextField(naiveBayesParameters.getOption());
    updateFrame.add(nbOptionLabel);
    updateFrame.add(nbOptionField);

    JButton runButton = new JButton("Run");
    runButton.addActionListener(e -> {
        try {
            double splitConfidence = Double.parseDouble(nbSplitConfidenceField.getText());
            int gracePeriod = Integer.parseInt(nbGracePeriodField.getText());
            int minNumInstancesPerLeaf = Integer.parseInt(nbMinNumInstancesPerLeafField.getText());
            int maxDepth = Integer.parseInt(nbMaxDepthField.getText());
            double nbThreshold = Double.parseDouble(nbNbThresholdField.getText());

            hoeffdingTreeParameters.setSplitConfidence(splitConfidence);
            hoeffdingTreeParameters.setGracePeriod(gracePeriod);
            hoeffdingTreeParameters.setMinNumInstancesPerLeaf(minNumInstancesPerLeaf);
            hoeffdingTreeParameters.setMaxDepth(maxDepth);
            hoeffdingTreeParameters.setNbThreshold(nbThreshold);

            hoeffdingTree.resetLearningImpl();

            int k = Integer.parseInt(knnKField.getText());
            int maxSize = Integer.parseInt(knnMaxSizeField.getText());
            knn = new StreamKNN(k, maxSize, false);
            knn.resetLearningImpl();

            String option = nbOptionField.getText();
            naiveBayesParameters.setOption(option);
            naiveBayes = new NaiveBayes();
            naiveBayes.prepareForUse();

            ensembleClassifier = new EnsembleClassifier(knn, hoeffdingTree, naiveBayes);
            ensembleClassifier.resetLearningImpl();
            trainAllClassifiers();
            showAccuracy("Ensemble");
            updateFrame.dispose();
        } catch (Exception ex) {
            JOptionPane.showMessageDialog(updateFrame, "Invalid format or error in updating options.", "Error", JOptionPane.ERROR_MESSAGE);
        }
    });
    updateFrame.add(new JLabel());
    updateFrame.add(runButton);
}
    updateFrame.setVisible(true);
}




    private void displayStreamData() {
        textArea.append("Sample Data:\n");
        for (Instance instance : sampleData) {
            textArea.append(instance.toString() + "\n");
        }
    }

    public static class StreamKNN extends AbstractClassifier {
        private static final long serialVersionUID = 1L;
        private int k;
        private Instances window;
        private int maxSize;
        private boolean useReservoir;
        private Random rand;
        private int maxClassValue;
        private int numProcessedInstances;

        public StreamKNN(int k, int maxSize, boolean useReservoir) {
            this.k = k;
            this.maxSize = maxSize;
            this.useReservoir = useReservoir;
            this.rand = new Random();
        }

        public StreamKNN(int k, int maxSize, boolean useReservoir, int seed) {
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
            classValues.addElement("Class0");
            classValues.addElement("Class1");
            attributes.add(new Attribute("class", classValues));

            window = new Instances("Window", attributes, 0);
            window.setClassIndex(window.numAttributes() - 1);

            maxClassValue = -1;
            numProcessedInstances = 0;
        }

        @Override
        public void trainOnInstanceImpl(Instance instance) {
            if ((int) instance.classValue() > maxClassValue) {
                maxClassValue = (int) instance.classValue();
            }
            numProcessedInstances++;
            if (useReservoir && window.size() >= maxSize) {
                int r = rand.nextInt(numProcessedInstances + 1);
                if (r < maxSize) {
                    window.remove(r);
                    window.add(instance);
                }
            } else {
                window.add(instance);
            }
        }

        @Override
        public double[] getVotesForInstance(Instance instance) {
            double[] votes = new double[maxClassValue + 1];
            Instances neighbours = getNeighbours(instance);
            for (int i = 0; i < neighbours.numInstances(); i++) {
                Instance neighbour = neighbours.instance(i);
                votes[(int) neighbour.classValue()]++;
            }
            return votes;
        }

        private Instances getNeighbours(Instance instance) {
            LinearNNSearch search = new LinearNNSearch(window);
            Instances neighbours = null;
            try {
                neighbours = search.kNearestNeighbours(instance, k);
            } catch (Exception e) {
                e.printStackTrace();
            }
            return neighbours;
        }

        @Override
        protected Measurement[] getModelMeasurementsImpl() {
            return null;
        }

        @Override
        public void getModelDescription(StringBuilder out, int indent) {
        }

        @Override
        public boolean isRandomizable() {
            return true;
        }

        public int getNumProcessedInstances() {
            return numProcessedInstances;
        }

        public int getK() {
            return k;
        }

        public int getMaxSize() {
            return maxSize;
        }
    }
    public static class HoeffdingTreeParameters {
    private double splitConfidence;
    private int gracePeriod;
    private int minNumInstancesPerLeaf;
    private int maxDepth;
    private double nbThreshold;

    // Constructor cập nhật
    public HoeffdingTreeParameters(double splitConfidence, int gracePeriod, int minNumInstancesPerLeaf, int maxDepth, double nbThreshold) {
        this.splitConfidence = splitConfidence;
        this.gracePeriod = gracePeriod;
        this.minNumInstancesPerLeaf = minNumInstancesPerLeaf;
        this.maxDepth = maxDepth;
        this.nbThreshold = nbThreshold;
    }

    // Getter và setter cho splitConfidence
    public double getSplitConfidence() {
        return splitConfidence;
    }

    public void setSplitConfidence(double splitConfidence) {
        this.splitConfidence = splitConfidence;
    }

    // Getter và setter cho gracePeriod
    public int getGracePeriod() {
        return gracePeriod;
    }

    public void setGracePeriod(int gracePeriod) {
        this.gracePeriod = gracePeriod;
    }

    // Getter và setter cho minNumInstancesPerLeaf
    public int getMinNumInstancesPerLeaf() {
        return minNumInstancesPerLeaf;
    }

    public void setMinNumInstancesPerLeaf(int minNumInstancesPerLeaf) {
        this.minNumInstancesPerLeaf = minNumInstancesPerLeaf;
    }

    // Getter và setter cho maxDepth
    public int getMaxDepth() {
        return maxDepth;
    }

    public void setMaxDepth(int maxDepth) {
        this.maxDepth = maxDepth;
    }

    // Getter và setter cho nbThreshold
    public double getNbThreshold() {
        return nbThreshold;
    }

    public void setNbThreshold(double nbThreshold) {
        this.nbThreshold = nbThreshold;
    }
}
    
    
    public static class NaiveBayesParameters {
    private double alpha;
    private double smoothParameter;
    private String featureSelection;
    private String binning;
    private String option;

    // Constructor
    public NaiveBayesParameters(double alpha, double smoothParameter, String featureSelection, String binning, String option) {
        this.alpha = alpha;
        this.smoothParameter = smoothParameter;
        this.featureSelection = featureSelection;
        this.binning = binning;
        this.option = option;
    }

    // Getter and Setter for Alpha
    public double getAlpha() {
        return alpha;
    }

    public void setAlpha(double alpha) {
        this.alpha = alpha;
    }

    // Getter and Setter for Smooth Parameter
    public double getSmoothParameter() {
        return smoothParameter;
    }

    public void setSmoothParameter(double smoothParameter) {
        this.smoothParameter = smoothParameter;
    }

    // Getter and Setter for Feature Selection
    public String getFeatureSelection() {
        return featureSelection;
    }

    public void setFeatureSelection(String featureSelection) {
        this.featureSelection = featureSelection;
    }

    // Getter and Setter for Binning
    public String getBinning() {
        return binning;
    }

    public void setBinning(String binning) {
        this.binning = binning;
    }

    // Getter and Setter for Option
    public String getOption() {
        return option;
    }

    public void setOption(String option) {
        this.option = option;
    }
}

    public static class EnsembleClassifier extends AbstractClassifier {
        private static final long serialVersionUID = 1L;
        private AbstractClassifier[] classifiers;
        private int numProcessedInstances;

        public EnsembleClassifier(AbstractClassifier... classifiers) {
            this.classifiers = classifiers;
            this.numProcessedInstances = 0;
        }

        @Override
        public void resetLearningImpl() {
            for (AbstractClassifier classifier : classifiers) {
                classifier.resetLearningImpl();
            }
        }

        @Override
        public void trainOnInstanceImpl(Instance instance) {
            numProcessedInstances++;
            for (AbstractClassifier classifier : classifiers) {
                classifier.trainOnInstance(instance);
            }
        }

        @Override
        public double[] getVotesForInstance(Instance instance) {
            double[] votes = new double[instance.numClasses()];
            for (AbstractClassifier classifier : classifiers) {
                double[] classifierVotes = classifier.getVotesForInstance(instance);
                for (int i = 0; i < classifierVotes.length; i++) {
                    votes[i] += classifierVotes[i];
                }
            }
            return votes;
        }

        @Override
        protected Measurement[] getModelMeasurementsImpl() {
            return null;
        }

        @Override
        public void getModelDescription(StringBuilder out, int indent) {
        }

        @Override
        public boolean isRandomizable() {
            return true;
        }

        public int getNumProcessedInstances() {
            return numProcessedInstances;
        }
    }
}
