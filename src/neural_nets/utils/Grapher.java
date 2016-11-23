package neural_nets.utils;

import java.awt.*;
import java.awt.geom.Ellipse2D;
import java.awt.geom.Rectangle2D;
import java.util.*;
import java.util.List;

import javafx.scene.shape.Circle;
import org.jfree.chart.ChartFactory;
import org.jfree.chart.ChartPanel;
import org.jfree.chart.JFreeChart;
import org.jfree.chart.axis.NumberAxis;
import org.jfree.chart.block.EmptyBlock;
import org.jfree.chart.plot.PlotOrientation;
import org.jfree.chart.plot.XYPlot;
import org.jfree.chart.renderer.xy.XYLineAndShapeRenderer;
import org.jfree.data.xy.XYDataset;
import org.jfree.data.xy.XYSeries;
import org.jfree.data.xy.XYSeriesCollection;
import org.jfree.ui.ApplicationFrame;
import org.jfree.ui.RefineryUtilities;
import org.nd4j.linalg.factory.Nd4j;

/**
 * A simple demonstration application showing how to create a line chart using data from an
 * {@link XYDataset}.
 *
 */
public class Grapher extends ApplicationFrame {
    private final JFreeChart chart;
    private Map<String,XYSeries> seriesMap;
    private XYSeriesCollection xyDataset;

    /**
     * Creates a new demo.
     *
     */
    public Grapher(double numEpochs) {

        super("Neural Network Visualization");
        this.seriesMap=new HashMap<>();
        xyDataset = initializeDataset();
        chart = createChart(xyDataset, numEpochs);
        final ChartPanel chartPanel = new ChartPanel(chart);
        chartPanel.setPreferredSize(new java.awt.Dimension(700, 540));
        setContentPane(chartPanel);
        pack();
        RefineryUtilities.centerFrameOnScreen(this);
        setVisible(true);
    }

    public void updateData(List<List<Double>> allErrors, List<List<Double>> allClassErrors, int currentEpoch) {

        List<Double> trainErrors = allErrors.get(0);
        List<Double> trainClassErrors = allClassErrors.get(0);
        List<Double> testErrors = allErrors.get(1);
        List<Double> testClassErrors = allClassErrors.get(1);
        List<Double> valErrors = allErrors.get(2);
        List<Double> valClassErrors = allClassErrors.get(2);

        if(trainErrors!=null) {
            createOrUpdateSeries("Training Set", trainErrors, currentEpoch);
        }
        if(trainClassErrors!=null) {
            createOrUpdateSeries("Training Set (Classification)", trainClassErrors, currentEpoch);
        }

        if(valErrors!=null) {
            createOrUpdateSeries("Validation Set", valErrors, currentEpoch);
        }
        if(valClassErrors!=null) {
            createOrUpdateSeries("Validation Set (Classification)",valClassErrors, currentEpoch);
        }

        if(testErrors!=null) {
            createOrUpdateSeries("Test Set", testErrors, currentEpoch);
        }
        if(testClassErrors!=null) {
            createOrUpdateSeries("Test Set (Classification)",testClassErrors, currentEpoch);
        }
    }

    /**
     * Creates a sample dataset.
     *
     * @return a sample dataset.
     */

    private XYSeries createOrUpdateSeries(String seriesTitle, List<Double> data, int currentEpoch) {
        final XYSeries series;
        if(seriesMap.containsKey(seriesTitle)) {
            series = seriesMap.get(seriesTitle);
        } else {
            series = new XYSeries(seriesTitle);
            seriesMap.put(seriesTitle,series);
            xyDataset.addSeries(series);
        }
        if(data.size() > series.getItemCount()) {
            double percentage = 1.0/(data.size()-series.getItemCount());
            int i = 0;
            for (Double point : data.subList(series.getItemCount(), data.size())) {
                series.add((double)currentEpoch + percentage*i, point, series.getItemCount()==data.size()-1);

                i++;
            }
        }
        return series;
    }

    private XYSeriesCollection initializeDataset() {
        final XYSeriesCollection dataset = new XYSeriesCollection();
        return dataset;
    }

    /**
     * Creates a chart.
     *
     * @return a chart.
     */
    private JFreeChart createChart(final XYDataset xyDataset, double numEpochs) {
        // create the chart...
        final JFreeChart chart = ChartFactory.createXYLineChart(
                "Error versus Learning Time",      // chart title
                "Epochs",                      // x axis label
                "Average Error",                      // y axis label
                xyDataset,                  // data
                PlotOrientation.VERTICAL,
                true,                     // include legend
                true,                     // tooltips
                false                     // urls
        );

        // NOW DO SOME OPTIONAL CUSTOMISATION OF THE CHART...
        chart.setBackgroundPaint(Color.white);

//        final StandardLegend legend = (StandardLegend) chart.getLegend();
        //      legend.setDisplaySeriesShapes(true);

        // get a reference to the plot for further customisation...
        final XYPlot plot = chart.getXYPlot();
        plot.setBackgroundPaint(Color.lightGray);
        //    plot.setAxisOffset(new Spacer(Spacer.ABSOLUTE, 5.0, 5.0, 5.0, 5.0));
        plot.setDomainGridlinePaint(Color.white);
        plot.setRangeGridlinePaint(Color.white);

        final XYLineAndShapeRenderer renderer = new XYLineAndShapeRenderer();
        renderer.setBaseShapesFilled(true);
        renderer.setBaseShapesVisible(true);
        renderer.setBaseLinesVisible(true);
        Rectangle rect = new Rectangle(2, 2);
        Rectangle large = new Rectangle(5,5);
        renderer.setSeriesLinesVisible(0,false);
        renderer.setSeriesLinesVisible(1,false);
        renderer.setSeriesShape(0, rect);
        renderer.setSeriesShape(1, rect);
        renderer.setSeriesShape(2, large);
        renderer.setSeriesShape(3, large);
        renderer.setSeriesShape(4, large);
        renderer.setSeriesShape(5, large);
        plot.setRenderer(renderer);

        // change the auto tick unit selection to integer units only...
        final NumberAxis rangeAxis = (NumberAxis) plot.getRangeAxis();
        rangeAxis.setStandardTickUnits(NumberAxis.createIntegerTickUnits());
        rangeAxis.setAutoRangeStickyZero(true);
        rangeAxis.setLowerBound(-0.01);
        // OPTIONAL CUSTOMISATION COMPLETED.

        final NumberAxis domainAxis = (NumberAxis) plot.getDomainAxis();
        domainAxis.setAutoRange(false);
        domainAxis.setAutoRangeStickyZero(true);
        domainAxis.setRange(0.0d,numEpochs);

        return chart;

    }

    // ****************************************************************************
    // * JFREECHART DEVELOPER GUIDE                                               *
    // * The JFreeChart Developer Guide, written by David Gilbert, is available   *
    // * to purchase from Object Refinery Limited:                                *
    // *                                                                          *
    // * http://www.object-refinery.com/jfreechart/guide.html                     *
    // *                                                                          *
    // * Sales are used to provide funding for the JFreeChart project - please    *
    // * support us so that we can continue developing free software.             *
    // ****************************************************************************

    /**
     * Starting point for the demonstration application.
     *
     * @param args ignored.
     */

}