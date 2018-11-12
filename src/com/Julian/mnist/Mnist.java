package com.Julian.mnist;

import java.io.File;

import data.TrainSet;
import functions.activation.ReLU;
import functions.activation.Softmax;
import functions.errorFunction.CrossEntropy;
import layers.ConvLayer;
import layers.DenseLayer;
import layers.PoolingLayer;
import layers.TransformationLayer;
import network.Network;
import network.NetworkBuilder;
import tools.ArrayTools;

/**
 * Mnist Convolutional Network
 * 
 * 07/25/18
 * 
 * @author Finn Eggers
 */


public class Mnist {

    public static void main(String[] args) {
    	
//		TrainSet trainSet = createTrainSet(0, 9999);
//
//		NetworkBuilder builder = new NetworkBuilder(1, 28, 28);
//		builder.addLayer(new TransformationLayer());
//		builder.addLayer(new DenseLayer(70).setActivationFunction(new Sigmoid()).biasRange(0, 1).weightsRange(-1, 1));
//		builder.addLayer(new DenseLayer(35).setActivationFunction(new Sigmoid()).biasRange(0, 1).weightsRange(-1, 1));
//		builder.addLayer(new DenseLayer(10).setActivationFunction(new Sigmoid()).biasRange(0, 1).weightsRange(-1, 1));
//		Network network = builder.buildNetwork();
//		network.setErrorFunction(new MeanSquaredError());
//				
//		network.train(trainSet, 100, 10, 0.3);
//
//		testTrainSet(network, trainSet, 10);
 


        NetworkBuilder builder = new NetworkBuilder(1, 28, 28);
        builder.addLayer(new ConvLayer(12, 5, 1, 2)
                .biasRange(0, 0)
                .weightsRange(-2, 2)
                .setActivationFunction(new ReLU()));
        builder.addLayer(new PoolingLayer(2));
        builder.addLayer(new ConvLayer(30, 5, 1, 0)
                .biasRange(0, 0)
                .weightsRange(-2, 2)
                .setActivationFunction(new ReLU()));
        builder.addLayer(new PoolingLayer(2));
        builder.addLayer(new TransformationLayer());
        builder.addLayer(new DenseLayer(120)
                .setActivationFunction(new ReLU())
        );
        builder.addLayer(new DenseLayer(10)
                .setActivationFunction(new Softmax())
        );
        Network network = builder.buildNetwork();
        network.setErrorFunction(new CrossEntropy());

        network.printArchitecture();

        TrainSet trainSet = createTrainSet(0, 9999);


        network.train(trainSet, 5, 10, 0.0003);

        testTrainSet(network, trainSet,1);

    }

    public static TrainSet createTrainSet(int start, int end) {
        TrainSet set = new TrainSet(1, 28, 28, 1, 1, 10);

        try {
            String path = new File("").getAbsolutePath();

            MnistImageFile m = new MnistImageFile(path + "/res/trainImage.idx3-ubyte", "rw");
            MnistLabelFile l = new MnistLabelFile(path + "/res/trainLabel.idx1-ubyte", "rw");

            for (int i = start; i <= end; i++) {
                if (i % 100 == 0) {
                    System.out.println("prepared: " + i);
                }

                double[][] input = new double[28][28];
                double[] output = new double[10];

                output[l.readLabel()] = 1d;
                for (int j = 0; j < 28 * 28; j++) {
                    input[j / 28][j % 28] = (double) m.read() / (double) 256;
                }

                set.addData(new double[][][]{input}, new double[][][]{{output}});
                m.next();
                l.next();
            }
            m.close();
            l.close();
        } catch (Exception e) {
            e.printStackTrace();
        }

        return set;
    }

    public static void trainData(Network net, TrainSet set, int epochs, int batch_size, double learningRate) {
        net.train(set, epochs, batch_size, learningRate);
    }

    public static void testTrainSet(Network net, TrainSet set, int printSteps) {
        int correct = 0;
        for (int i = 0; i < set.size(); i++) {

            double highest = ArrayTools.indexOfHighestValue(ArrayTools.convertFlattenedArray(net.calculate(set.getInput(i))));
            double actualHighest = ArrayTools.indexOfHighestValue(ArrayTools.convertFlattenedArray(set.getOutput(i)));
            if (highest == actualHighest) {
                correct++;
            }
            if (i % printSteps == 0) {
                System.out.println(i + ": " + (double) correct / (double) (i + 1));
            }
        }
        System.out.println("Testing finished, RESULT: " + correct + " / " + set.size() + "  -> " + ((double) correct / (double) set.size()) * 100 + " %");
    }
}
