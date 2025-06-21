package jp.jaxa.iss.kibo.rpc.sampleapk;

import android.util.Log;

import gov.nasa.arc.astrobee.Kinematics;
import gov.nasa.arc.astrobee.Result;
import jp.jaxa.iss.kibo.rpc.api.KiboRpcService;

import gov.nasa.arc.astrobee.types.Point;
import gov.nasa.arc.astrobee.types.Quaternion;

import org.opencv.aruco.Aruco;
import org.opencv.aruco.Dictionary;
import org.opencv.calib3d.Calib3d;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.Rect;
import org.opencv.core.Scalar;
import org.opencv.imgproc.Imgproc;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * Class meant to handle commands from the Ground Data System and execute them in Astrobee.
 */

public class YourService extends KiboRpcService {
    private final String TAG = this.getClass().getSimpleName();
    private final Map<Integer, Point> points = new HashMap<>();
    private final Map<Integer, Quaternion> orientations = new HashMap<>();
    @Override
    protected void runPlan1() {
        api.startMission();
        Log.i(TAG, "Mission started");
        points.put(1, new Point(10.66, -9.79, 4.905));
        points.put(2, new Point(10.91, -8.875, 4.55));
        points.put(3, new Point(10.535, -7.88, 4.55));
        points.put(4, new Point(10.66, -6.8525, 4.95));
        // Astronaut point id 10
        points.put(10, new Point(11.005, -6.808, 4.965));

        orientations.put(1, new Quaternion(-0.009f, -0.224f, -0.455f, 0.862f));
        orientations.put(2, new Quaternion(0f, 0.707f, 0f, 0.707f));
        orientations.put(3, new Quaternion(0f, 0.609f, 0f, 0.793f));
        orientations.put(4, new Quaternion(0.009f, -0.001f, -0.996f, 0.087f));
        orientations.put(10, new Quaternion(0f, 0f, -0.707f, 0.707f));
        // Move to Area 1
        for (int areaId = 1; areaId <= 4; areaId++) {
            Point target = points.get(areaId);
            Quaternion orient = orientations.get(areaId);
            Log.i(TAG, "Moving to area " + areaId + ": " + target);
            boolean arrived = moveToWrapper(target, orient);
            Kinematics kin = api.getRobotKinematics();
            Point current = kin.getPosition();
            double dist = Math.sqrt(Math.pow(current.getX() - target.getX(), 2)
                    + Math.pow(current.getY() - target.getY(), 2)
                    + Math.pow(current.getZ() - target.getZ(), 2));
            Log.i(TAG, String.format("Arrived at area %d. Current pos: (%.3f,%.3f,%.3f). Distance to goal: %.3fm",
                    areaId, current.getX(), current.getY(), current.getZ(), dist));

            // --- If more than 0.08m (8cm) from the goal, move closer ---
            if (dist > 0.08) {
                Log.i(TAG, "Nudging toward target for finer alignment...");
                // Move a fraction closer
                double fx = target.getX() - current.getX();
                double fy = target.getY() - current.getY();
                double fz = target.getZ() - current.getZ();
                Point nudge = new Point(current.getX() + fx * 0.7,
                        current.getY() + fy * 0.7,
                        current.getZ() + fz * 0.7);
                moveToWrapper(nudge, orient);

            }
            CroppedARTag tag = cropFirstARTagFromNavCam();
            if (tag == null) {
                Log.w(TAG, "No AR tag found at area " + areaId + ". Trying a small sideways move.");
                // Try nudging sideways by 5 cm (along y or z, e.g.)
                Point smallNudge = new Point(current.getX(), current.getY() + 0.05, current.getZ());
                moveToWrapper(smallNudge, orient);

                // Try AR detection again
                tag = cropFirstARTagFromNavCam();
                if (tag == null) {
                    Log.e(TAG, "Still no AR tag after adjustment at area " + areaId + ". Skipping to next area.");
                } else {
                    Log.i(TAG, "AR tag found after adjustment at area " + areaId + "!");
                }
            } else {
                Log.i(TAG, "AR tag found at area " + areaId + "!");
            }

        }
        Point p1 = new Point(11.143d, -6.7607d, 4.9654d);
        Quaternion q1 = new Quaternion(0f, 0f, 0.707f, 0.707f);
        api.moveTo(p1, q1, false);
        api.reportRoundingCompletion();
        api.notifyRecognitionItem();
        api.takeTargetItemSnapshot();


    }

    @Override
    protected void runPlan2(){
        // write your plan 2 here.
    }

    @Override
    protected void runPlan3(){
        // write your plan 3 here.
    }
    //Check
    private boolean moveToWrapper(Point goal, Quaternion orientation) {
        final int MAX_RETRY = 3;
        Result result = null;

        for (int attempt = 1; attempt <= MAX_RETRY; attempt++) {
            Log.i(TAG, "moveToWrapper: attempt " + attempt
                    + " → goal=" + goal + ", orientation=" + orientation);
            result = api.moveTo(goal, orientation, false);
            if (result.hasSucceeded()) {
                Log.i(TAG, "moveToWrapper: succeeded on attempt " + attempt);
                return true;
            } else {
                Log.w(TAG, "moveToWrapper: failed attempt " + attempt);
            }
        }

        Log.e(TAG, "moveToWrapper: all " + MAX_RETRY + " attempts failed");
        return true;
    }
    public class CroppedARTag {
        public final int id;
        public final Mat cropped;
        public final double[] center;
        public CroppedARTag(int id, Mat cropped, double[] center) {
            this.id = id;
            this.cropped = cropped;
            this.center = center;
        }
    }

    /**
     * Detect the first ArUco tag in a NavCam image, crop a region around it, and log details.
     * @return CroppedARTag or null if none found.
     */
    private CroppedARTag cropFirstARTagFromNavCam() {
        // 1. Get and undistort the NavCam image
        double[][] cameraParam = api.getNavCamIntrinsics();
        Mat cameraMatrix = new Mat(3, 3, CvType.CV_64F);
        Mat distCoeffs   = new Mat(1, 5, CvType.CV_64F);
        cameraMatrix.put(0, 0, cameraParam[0]);
        distCoeffs.put(0, 0, cameraParam[1]);

        Mat src = api.getMatNavCam();
        Mat color = new Mat();
        Imgproc.cvtColor(src, color, Imgproc.COLOR_GRAY2BGR);

        Mat undistorted = new Mat();
        Calib3d.undistort(src, undistorted, cameraMatrix, distCoeffs);

        // 2. Detect markers
        Dictionary arucoDict = Aruco.getPredefinedDictionary(Aruco.DICT_5X5_250);
        Mat ids = new Mat();
        List<Mat> corners = new ArrayList<>();
        Aruco.detectMarkers(undistorted, arucoDict, corners, ids);

        if (corners.isEmpty() || ids.empty()) {
            Log.i(TAG, "No AR tag detected.");
            return null;
        }

        // 3. Pick the first tag found
        Mat firstCorner = corners.get(0);
        int tagId = (int) ids.get(0, 0)[0];

        // Get the 4 corners
        double[] tl = firstCorner.get(0, 0);
        double[] tr = firstCorner.get(0, 1);
        double[] br = firstCorner.get(0, 2);
        double[] bl = firstCorner.get(0, 3);

        // Center in pixel coordinates
        double centerX = (tl[0] + tr[0] + br[0] + bl[0]) / 4.0;
        double centerY = (tl[1] + tr[1] + br[1] + bl[1]) / 4.0;
        double[] center = { centerX, centerY };

        Log.i(TAG, String.format("AR tag ID: %d", tagId));
        Log.i(TAG, String.format("Corners: TL(%.1f,%.1f) TR(%.1f,%.1f) BR(%.1f,%.1f) BL(%.1f,%.1f)",
                tl[0], tl[1], tr[0], tr[1], br[0], br[1], bl[0], bl[1]));
        Log.i(TAG, String.format("Center: (%.1f, %.1f)", centerX, centerY));

        // 4. Compute crop rectangle (expand a little around tag)
        int imgW = undistorted.cols(), imgH = undistorted.rows();
        double tagSide = (   Math.hypot(tl[0] - tr[0], tl[1] - tr[1])
                + Math.hypot(tr[0] - br[0], tr[1] - br[1])
                + Math.hypot(br[0] - bl[0], br[1] - bl[1])
                + Math.hypot(bl[0] - tl[0], bl[1] - tl[1])
        ) / 4.0;
        int margin = (int)(tagSide * 0.5);
        int xMin = (int)Math.max(0, centerX - tagSide/2 - margin);
        int yMin = (int)Math.max(0, centerY - tagSide/2 - margin);
        int xMax = (int)Math.min(imgW, centerX + tagSide/2 + margin);
        int yMax = (int)Math.min(imgH, centerY + tagSide/2 + margin);

        Rect roi = new Rect(xMin, yMin, xMax - xMin, yMax - yMin);
        Mat cropped = new Mat(undistorted, roi);

        // 5. Save for review
        api.saveMatImage(cropped, "ar_crop_" + tagId + ".png");
        Log.i(TAG, String.format("Saved cropped AR tag as ar_crop_%d.png", tagId));

        return new CroppedARTag(tagId, cropped, center);
    }

}


