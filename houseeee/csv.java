import weka.classifiers.trees.RandomForest;
import weka.core.Instances;
import weka.core.converters.CSVLoader;
import weka.classifiers.evaluation.Evaluation;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Remove;
import java.io.File;
import java.io.FileWriter;

public class csv {
    public static void main(String[] args) {
        // Path to your CSV files for training and testing
        String trainCsvFilePath = "train.csv";
        String testCsvFilePath = "test.csv";
        String resultCsvFilePath = "testresult.csv"; // Path to save the test results

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

            // Create FileWriter to write to CSV file
            FileWriter writer = new FileWriter(resultCsvFilePath);

            // Write header to CSV file
            StringBuilder header = new StringBuilder();
            for (int i = 0; i < testData.numAttributes(); i++) {
                header.append(testData.attribute(i).name()).append(",");
            }
            header.append("ActualPrice,PredictedPrice\n");
            writer.append(header.toString());

            // Loop through test instances to make predictions and evaluate
            for (int i = 0; i < testData.numInstances(); i++) {
                double actualPrice = testData.instance(i).classValue();
                double predictedPrice = rfModel.classifyInstance(testData.instance(i));
                testEval.evaluateModelOnceAndRecordPrediction(rfModel, testData.instance(i));

                // Write instance values, actual price, and predicted price to CSV file
                StringBuilder instanceValues = new StringBuilder();
                for (int j = 0; j < testData.numAttributes(); j++) {
                    instanceValues.append(testData.instance(i).value(j)).append(",");
                }
                instanceValues.append(actualPrice).append(",").append(predictedPrice).append("\n");
                writer.append(instanceValues.toString());
            }

            // Close the FileWriter
            writer.flush();
            writer.close();

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

