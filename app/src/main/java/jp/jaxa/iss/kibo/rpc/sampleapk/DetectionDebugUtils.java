package jp.jaxa.iss.kibo.rpc.sampleapk;

import android.graphics.Bitmap;
import android.util.Log;
import org.tensorflow.lite.DataType;
import org.tensorflow.lite.Interpreter;
import org.tensorflow.lite.support.image.TensorImage;
import java.util.ArrayList;
import java.util.List;

public class DetectionDebugUtils {

    private static final String TAG = "DetectionDebugUtils";

    public static class ClassificationResult {
        public final String label;
        public final float confidence;
        public ClassificationResult(String label, float confidence) {
            this.label = label;
            this.confidence = confidence;
        }
    }

    public static ClassificationResult classifyImageWithConfidence(Bitmap bitmap, Interpreter tflite, List<String> labels) {
        Bitmap resized = Bitmap.createScaledBitmap(bitmap, 224, 224, true);
        TensorImage tensorImage = new TensorImage(DataType.FLOAT32);
        tensorImage.load(resized);

        float[][] output = new float[1][labels.size()];
        tflite.run(tensorImage.getBuffer(), output);

        int maxIdx = 0;
        float maxScore = output[0][0];
        for (int i = 1; i < output[0].length; i++) {
            if (output[0][i] > maxScore) {
                maxIdx = i;
                maxScore = output[0][i];
            }
        }
        String predictedLabel = labels.get(maxIdx);
        Log.i(TAG, "Classification: " + predictedLabel + " (" + maxScore + ")");
        return new ClassificationResult(predictedLabel, maxScore);
    }

    public static class DetectionResult {
        public final String label;
        public final float confidence;
        public final float xmin, ymin, xmax, ymax;
        public DetectionResult(String label, float confidence, float xmin, float ymin, float xmax, float ymax) {
            this.label = label;
            this.confidence = confidence;
            this.xmin = xmin;
            this.ymin = ymin;
            this.xmax = xmax;
            this.ymax = ymax;
        }
    }

    public static List<DetectionResult> detectObjectsWithResults(Bitmap bitmap, Interpreter tflite, List<String> labels) {
        Bitmap resized = Bitmap.createScaledBitmap(bitmap, 320, 320, true);
        TensorImage tensorImage = new TensorImage(DataType.FLOAT32);
        tensorImage.load(resized);

        float[][] output = new float[10][6];
        tflite.run(tensorImage.getBuffer(), output);

        List<DetectionResult> detections = new ArrayList<>();
        for (int i = 0; i < output.length; i++) {
            float score = output[i][5];
            if (score < 0.5) continue;
            int classIdx = (int) output[i][4];
            String label = labels.get(classIdx);
            detections.add(new DetectionResult(
                    label, score,
                    output[i][1], output[i][0], output[i][3], output[i][2]
            ));
            Log.i(TAG, String.format(
                    "Detection %d: %s (%.2f) at [%.2f, %.2f, %.2f, %.2f]",
                    i, label, score, output[i][1], output[i][0], output[i][3], output[i][2]
            ));
        }
        return detections;
    }
}
