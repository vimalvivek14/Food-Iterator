/*
 * Copyright (c) 2019 Skymind AI Bhd.
 * Copyright (c) 2020 CertifAI Sdn. Bhd.
 *
 * This program and the accompanying materials are made available under the
 * terms of the Apache License, Version 2.0 which is available at
 * https://www.apache.org/licenses/LICENSE-2.0.
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 * License for the specific language governing permissions and limitations
 * under the License.
 *
 * SPDX-License-Identifier: Apache-2.0
 */
package ai.certifai.solution;

import com.fatsecret.platform.model.*;
import com.fatsecret.platform.services.Response;
import com.fatsecret.platform.utils.FoodUtility;
import com.fatsecret.platform.utils.ServingUtility;
import org.json.JSONArray;
import org.json.JSONException;
import org.json.JSONObject;
import org.bytedeco.javacv.CanvasFrame;
import org.bytedeco.javacv.Frame;
import org.bytedeco.javacv.FrameGrabber;
import org.bytedeco.javacv.OpenCVFrameConverter;
import org.bytedeco.opencv.opencv_core.Mat;
import org.bytedeco.opencv.opencv_core.Size;
import org.datavec.image.loader.NativeImageLoader;
import org.datavec.image.transform.ColorConversionTransform;
import org.deeplearning4j.api.storage.StatsStorage;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.BackpropType;
import org.deeplearning4j.nn.conf.GradientNormalization;
import org.deeplearning4j.nn.conf.WorkspaceMode;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.transferlearning.FineTuneConfiguration;
import org.deeplearning4j.nn.transferlearning.TransferLearning;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.deeplearning4j.ui.api.UIServer;
import org.deeplearning4j.ui.stats.StatsListener;
import org.deeplearning4j.ui.storage.InMemoryStatsStorage;
import org.deeplearning4j.util.ModelSerializer;
import org.deeplearning4j.zoo.PretrainedType;
import org.deeplearning4j.zoo.model.VGG16;
import org.nd4j.evaluation.classification.Evaluation;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.preprocessor.VGG16ImagePreProcessor;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.learning.config.Sgd;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import javax.swing.*;
import java.awt.event.ActionEvent;
import java.awt.event.KeyEvent;
import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import com.fatsecret.platform.services.FatsecretService;

import static org.bytedeco.opencv.global.opencv_core.flip;
import static org.bytedeco.opencv.global.opencv_imgproc.*;

public class CapstoneFood {
    private static final Logger log = LoggerFactory.getLogger(CapstoneFood.class);
    private static int seed = 123;

    private static int batchSize = 10;
    private static int nEpochs = 1;
    private static double learningRate = 0.1;
    private static int nClasses = 11;
    private static List<String> labels;

    private static File modelFilename = new File(System.getProperty("user.dir"), "generated-models/CapstoneFood.zip");
    private static ComputationGraph model;
    private static Frame frame = null;
    private static final String key = "INSERT KEY";
    private static final String secret = "INSERT SECRET";

    public static void main(String[] args) throws Exception {

        // Directory for Custom train and test datasets
        log.info("Load data...");
        FoodIterator.setup(batchSize,70);
        RecordReaderDataSetIterator trainIter = FoodIterator.trainIterator(batchSize);
        trainIter.setPreProcessor( new VGG16ImagePreProcessor());

        RecordReaderDataSetIterator testIter = FoodIterator.testIterator(batchSize);
        testIter.setPreProcessor( new VGG16ImagePreProcessor());

        // Print Labels
        labels = trainIter.getLabels();
        //System.out.println(Arrays.toString(labels.toArray()));

        if (modelFilename.exists()) {

            // Load trained model from previous execution
            log.info("Load model...");
            model = ModelSerializer.restoreComputationGraph(modelFilename);
            log.info(model.summary());

        } else {

            log.info("Build model...");
            // Load pretrained VGG16 model
            ComputationGraph pretrained = (ComputationGraph) VGG16.builder().build().initPretrained(PretrainedType.IMAGENET);
            log.info(pretrained.summary());

            // Transfer Learning steps - Model Configurations.
            FineTuneConfiguration fineTuneConf = getFineTuneConfiguration();

            // Transfer Learning steps - Modify prebuilt model's architecture for current scenario
            model = buildComputationGraph(pretrained, fineTuneConf);

            log.info("Train model...");

            UIServer server = UIServer.getInstance();
            StatsStorage storage = new InMemoryStatsStorage();
            server.attach(storage);
            model.setListeners(new ScoreIterationListener(1), new StatsListener(storage));

            for (int i = 0; i < nEpochs; i++) {
                trainIter.reset();
                while (trainIter.hasNext()) {
                    model.fit(trainIter.next());
                }
                log.info("*** Completed epoch {} ***", i);
            }

            ModelSerializer.writeModel(model, modelFilename, true);

            //validationTestDataset(testIter);
            Evaluation eval = model.evaluate(testIter);
            log.info("Train Accuracy: " + model.evaluate(trainIter).accuracy());
            log.info("Test Accuracy: " + eval.accuracy());
            log.info(eval.stats());
        }

        System.out.println("Starting Webcam");
        //Using model with camera
        doInference();
    }
    private static ComputationGraph buildComputationGraph(ComputationGraph pretrained, FineTuneConfiguration fineTuneConf) {
        //Construct a new model with the intended architecture and print summary
        ComputationGraph vgg16Transfer = new TransferLearning.GraphBuilder(pretrained)
                .fineTuneConfiguration(fineTuneConf)
                .setFeatureExtractor("fc1") //"fc1" and below are frozen
                .removeVertexKeepConnections("predictions") //replace the functionality of the final vertex
                .addLayer("newpredictions",new OutputLayer
                        .Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                        .activation(Activation.SOFTMAX)
                        .nIn(4096)
                        .weightInit(WeightInit.XAVIER)
                        .nOut(nClasses)
                        .build(),"fc2") //add in a final output dense layer,
                // configurations on a new layer here will be override the finetune confs.
                // For eg. activation function will be softmax not RELU
                .setOutputs("newpredictions") //since we removed the output vertex and it's connections we need to specify outputs for the graph
                .build();
        log.info(vgg16Transfer.summary());

        return vgg16Transfer;
    }

    private static FineTuneConfiguration getFineTuneConfiguration() {

        FineTuneConfiguration _FineTuneConfiguration = new FineTuneConfiguration.Builder()
                .seed(seed)
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .gradientNormalization(GradientNormalization.RenormalizeL2PerLayer)
                .gradientNormalizationThreshold(1.0)
                .updater(new Sgd.Builder().learningRate(learningRate).build())
                .l2(0.00001)
                .activation(Activation.LEAKYRELU)
                .backpropType(BackpropType.Standard)
                .trainingWorkspaceMode(WorkspaceMode.ENABLED)
                .inferenceWorkspaceMode(WorkspaceMode.ENABLED)
                .build();

        return _FineTuneConfiguration;
    }

    private static void getPredictions(INDArray image) throws IOException {
        INDArray[] output = model.output(false, image);
        List<CapstoneFood.Prediction> predictions = decodePredictions(output[0], nClasses);
        System.out.println("==================================================================================================================================");
        System.out.println("Prediction: "+predictions.get(0).toString());
        searchFoodItems(predictions.get(0).toString());
    }


    private static List<CapstoneFood.Prediction> decodePredictions(INDArray encodedPredictions, int numPredicted) throws IOException {
        List<CapstoneFood.Prediction> decodedPredictions = new ArrayList<>();
        int[] topX = new int[numPredicted];
        float[] topXProb = new float[numPredicted];

        int i = 0;
        for (INDArray currentBatch = encodedPredictions.getRow(0).dup(); i < numPredicted; ++i) {

            topX[i] = Nd4j.argMax(currentBatch, 1).getInt(0);
            topXProb[i] = currentBatch.getFloat(0, topX[i]);
            currentBatch.putScalar(0, topX[i], 0.0D);
            decodedPredictions.add(new CapstoneFood.Prediction(labels.get(topX[i])));
        }
        return decodedPredictions;
    }

    public static class Prediction {

        private String label;

        public Prediction(String label) {
            this.label = label;
        }

        public void setLabel(final String label) {
            this.label = label;
        }

        public String toString() {
            return String.format("%s ", this.label);
        }
    }



    //
    private static void doInference() {

        String cameraPos = "front";
        int cameraNum = 0;
        Thread thread = null;
        NativeImageLoader loader = new NativeImageLoader(
                224,
                224,
                3,
                new ColorConversionTransform(COLOR_BGR2RGB));
        VGG16ImagePreProcessor scaler = new VGG16ImagePreProcessor();
        if (!cameraPos.equals("front") && !cameraPos.equals("back")) {
            try {
                throw new Exception("Unknown argument for camera position. Choose between front and back");
            } catch (Exception e) {
                e.printStackTrace();
            }
        }

        FrameGrabber grabber = null;
        try {
            grabber = FrameGrabber.createDefault(cameraNum);
        } catch (FrameGrabber.Exception e) {
            e.printStackTrace();
        }
        OpenCVFrameConverter.ToMat converter = new OpenCVFrameConverter.ToMat();

        try {
            grabber.start();
        } catch (FrameGrabber.Exception e) {
            e.printStackTrace();
        }

        CanvasFrame canvas = new CanvasFrame("Press 'Enter' for Details: ");
        int w = grabber.getImageWidth();
        int h = grabber.getImageHeight();
        canvas.setCanvasSize(w, h);
        canvas.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
        final JButton b=new JButton("button");
        canvas.add(b);
        canvas.getRootPane().getInputMap(JComponent.WHEN_IN_FOCUSED_WINDOW).put(KeyStroke.getKeyStroke(KeyEvent.VK_ENTER,0),"clickButton");



        while (true) {

            try {
                frame = grabber.grab();
            } catch (FrameGrabber.Exception e) {
                e.printStackTrace();
            }

            //if a thread is null, create new thread
            if (thread == null) {
                thread = new Thread(() ->
                {
                    while (frame != null) {
                        try {
                            Mat rawImage = new Mat();

                            //Flip the camera if opening front camera
                            if (cameraPos.equals("front")) {
                                Mat inputImage = converter.convert(frame);
                                flip(inputImage, rawImage, 1);
                            } else {
                                rawImage = converter.convert(frame);
                            }

                            Mat resizeImage = new Mat();
                            resize(rawImage, resizeImage, new Size(FoodIterator.width, FoodIterator.height));
                            INDArray inputImage = loader.asMatrix(resizeImage);
                            scaler.transform(inputImage);
                            //getPredictions(inputImage);
                            canvas.showImage(converter.convert(rawImage));
                            canvas.getRootPane().getActionMap().put("clickButton",new AbstractAction(){
                                public void actionPerformed(ActionEvent ae)
                                {
                                    b.doClick();
                                    try {
                                        getPredictions(inputImage);
                                    } catch (IOException e) {
                                        e.printStackTrace();
                                    }
                                }
                            });
                        } catch (Exception e) {
                            throw new RuntimeException(e);
                        }
                    }
                });
                thread.start();
            }

            KeyEvent t = null;
            try {
                t = canvas.waitKey(33);
            } catch (InterruptedException e) {
                e.printStackTrace();
            }

            if ((t != null) && (t.getKeyCode() == KeyEvent.VK_Q)) {
                break;
            }
        }
    }
    private static void searchFoodItems(String name) {
       try {
           FatsecretService serviceCall = new FatsecretService(key, secret);
           name=name.replaceAll("_", "\\s+").toLowerCase();
           Response<CompactFood> responseAnswer = serviceCall.searchFoods(name);
           if (responseAnswer == null) {
               System.out.println("Not in Database");
           }
           //This response contains the list of food items at zeroth page for your query
           List<CompactFood> results = responseAnswer.getResults();
           //This list contains summary information about the food items
           Long id = results.get(0).getId();

           Food food = serviceCall.getFood(id);
           //This food object contains detailed information about the food item
           System.out.println("Name: "+food.getName());
           System.out.println("Calories: "+food.getServings().get(0).getCalories()+"kcal/"+food.getServings().get(0).getMeasurementDescription());
           System.out.println("Serving: "+food.getServings().get(0).getMetricServingAmount()+food.getServings().get(0).getMetricServingUnit());
           System.out.println("Protein: "+food.getServings().get(0).getProtein()+"g/serving");
           System.out.println("Carbohydrate: "+food.getServings().get(0).getCarbohydrate()+"g/serving");
           System.out.println("Fat: "+food.getServings().get(0).getFat()+"g/serving");
       }catch(Exception e){System.out.println("Information Unavailable");}
    }



}
