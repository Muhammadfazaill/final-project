import weka.classifiers.trees.RandomForest;
import weka.core.Instances;
import weka.core.converters.CSVLoader;
import weka.classifiers.evaluation.Evaluation;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Remove;
import java.io.File;


public class testdataprediction {
    public static void main(String[] args) {
        // Path to your CSV files for training and testing
        String trainCsvFilePath = "train.csv";
        String testCsvFilePath = "test.csv";

        try {
            // Load training CSV file
            CSVLoader trainLoader = new CSVLoader();
            trainLoader.setFile(new File(trainCsvFilePath));
            Instances trainData = trainLoader.getDataSet();

            // Remove the "size_units" attribute from training data
            String[] options = {"-R", String.valueOf(trainData.attribute("size_units").index() + 1)};
            Remove removeFilter = new Remove();
            removeFilter.setOptions(options);
            removeFilter.setInputFormat(trainData);
            trainData = Filter.useFilter(trainData, removeFilter);

            // Set class attribute (price) index for training data
            trainData.setClassIndex(trainData.numAttributes() - 1);

            // Create and build RandomForest model
            RandomForest rfModel = new RandomForest(); // Instantiate RandomForest classifier
            rfModel.buildClassifier(trainData);

            // Load test CSV file
            CSVLoader testLoader = new CSVLoader();
            testLoader.setFile(new File(testCsvFilePath));
            Instances testData = testLoader.getDataSet();

            // Remove the "size_units" attribute from test data (assuming it was removed from training data)
            String[] testOptions = {"-R", String.valueOf(testData.attribute("size_units").index() + 1)};
            Remove testRemoveFilter = new Remove();
            testRemoveFilter.setOptions(testOptions);
            testRemoveFilter.setInputFormat(testData);
            testData = Filter.useFilter(testData, testRemoveFilter);

            // Set class attribute (price) index for test data
            testData.setClassIndex(testData.numAttributes() - 1);

            // Create Evaluation object for test data
            Evaluation testEval = new Evaluation(testData);

            // Loop through test instances to make predictions and evaluate
            for (int i = 0; i < testData.numInstances(); i++) {
                double actualPrice = testData.instance(i).classValue();
                double predictedPrice = rfModel.classifyInstance(testData.instance(i));
                testEval.evaluateModelOnceAndRecordPrediction(rfModel, testData.instance(i));
            }

            // Print test evaluation results
            System.out.println("\nTest Evaluation Results:");
            System.out.println("Mean absolute error: " + testEval.meanAbsoluteError());
            System.out.println("Root mean squared error: " + testEval.rootMeanSquaredError());
            System.out.println("Coefficient of determination (R^2): " + testEval.correlationCoefficient());

        } catch (Exception e) {
            e.printStackTrace();
        }
    }
}

