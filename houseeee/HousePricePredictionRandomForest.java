import weka.classifiers.trees.RandomForest;
import weka.core.Instances;
import weka.core.converters.CSVLoader;
import weka.classifiers.evaluation.Evaluation;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Remove;
import java.io.File;
import org.jfree.chart.ChartFactory;
import org.jfree.chart.ChartPanel;
import org.jfree.chart.JFreeChart;
import org.jfree.chart.plot.XYPlot;
import org.jfree.data.xy.XYSeries;
import org.jfree.data.xy.XYSeriesCollection;
import javax.swing.*;
import java.awt.*;

public class HousePricePredictionRandomForest {
    public static void main(String[] args) {
        // Path to your CSV file
        String csvFilePath = "train.csv";

        try {
            // Load CSV file
            CSVLoader loader = new CSVLoader();
            loader.setFile(new File(csvFilePath));
            Instances data = loader.getDataSet();

            // Remove the "size_units" attribute
            String[] options = {"-R", String.valueOf(data.attribute("size_units").index() + 1)};
            Remove removeFilter = new Remove();
            removeFilter.setOptions(options);
            removeFilter.setInputFormat(data);
            data = Filter.useFilter(data, removeFilter);

            // Set class attribute (price) index
            data.setClassIndex(data.numAttributes() - 1);

            // Create and build RandomForest model
            RandomForest rfModel = new RandomForest(); // Instantiate RandomForest classifier
            rfModel.buildClassifier(data);

            // Evaluate RandomForest model
            Evaluation rfEval = new Evaluation(data);
            rfEval.evaluateModel(rfModel, data);

            // Create series for scatter plot
            XYSeries actualSeries = new XYSeries("Actual Prices");
            XYSeries predictedSeries = new XYSeries("Predicted Prices");

            // Populate scatter series with actual vs predicted prices
            for (int i = 0; i < data.numInstances(); i++) {
                double actualPrice = data.instance(i).classValue();
                double predictedPrice = rfModel.classifyInstance(data.instance(i));
                actualSeries.add(actualPrice, actualPrice); // Use the same value for x and y for actual prices
                predictedSeries.add(actualPrice, predictedPrice); // Use the actual price for x and predicted price for y
            }

            // Create dataset for scatter plot
            XYSeriesCollection dataset = new XYSeriesCollection();
            dataset.addSeries(actualSeries);
            dataset.addSeries(predictedSeries);

            // Create scatter plot
            JFreeChart scatterChart = ChartFactory.createScatterPlot("Actual vs Predicted Prices", "Actual Price", "Predicted Price", dataset);

            // Set colors for actual and predicted series
            XYPlot plot = (XYPlot) scatterChart.getPlot();
            plot.getRenderer().setSeriesPaint(0, Color.ORANGE); // Actual Prices - Orange
            plot.getRenderer().setSeriesPaint(1, Color.BLUE); // Predicted Prices - Blue

            // Create frame to display scatter plot
            JFrame scatterFrame = new JFrame("Scatter Plot");
            scatterFrame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
            scatterFrame.add(new ChartPanel(scatterChart));
            scatterFrame.pack();
            scatterFrame.setVisible(true);

            // Print RandomForest evaluation results
            System.out.println("\nRandomForest Evaluation Results:");
            System.out.println("Mean absolute error: " + rfEval.meanAbsoluteError());
            System.out.println("Root mean squared error: " + rfEval.rootMeanSquaredError());
            System.out.println("Coefficient of determination (R^2): " + rfEval.correlationCoefficient());

        } catch (Exception e) {
            e.printStackTrace();
        }
    }
}






