# 1. Task 1s

## 1.1 Navigation

For navigation, we defined several keypoints on the map. The robot follows these keypoints in a loop, which allows it to repeat the route if it does not detect all faces during the first pass.

## 1.2 Face Detection

For face detection, we used the provided YOLOv8n model. Detection worked well in simulation, but we added several improvements to make detections and marker positions more robust:

- **Detection threshold:** detections from large distances are ignored because they have a higher chance of being false positives.
- **Deduplication radius:** since faces are not placed close together, detections that are only slightly offset from one another are treated as the same face.
- **Approach offset:** the robot does not drive directly into the face marker. Instead, it navigates to a position slightly in front of the marker before greeting the person.
- **Median position:** every valid detection of the same face is stored in an array. Before creating the final marker, we take the median of these positions. We chose the median instead of the mean because some detections can fall on the edge of a nearby box, which would strongly affect the mean and produce an incorrect marker position.

## 1.3 Ring Detection

### 1.3.1 Detection

Ring detection was implemented using OpenCV's `HoughCircles` function, which performs Hough transform circle detection on the depth image. Using depth instead of RGB reduced the chance of misclassifying 2D ring stickers on boxes as real rings.

After detecting a circle, we used the point cloud callback to get a 3D point on the ring, offset by the detected radius. This effectively gave us the center of the ring. The remaining robustness tricks were similar to those used for face detection.

### 1.3.2 Color Classification

For color classification, we captured several images of the rings from different angles. From these images, we built color histograms for each ring color.

**TODO:** Gasper: describe how the histograms are generated.

We averaged all histograms for each color. The resulting average histograms were then used for classification. When a ring was detected, we compared its histogram to the stored color histograms and incremented the counter for the most similar color in a dictionary of possible colors.

We also used Lowe's ratio test to make sure that the difference between the best and second-best matches was large enough. After a certain number of detections, the ring was assigned the color with the highest number of classifications.

We had some issues with color detection because the stick holding the ring was also sometimes included in the detection. This changed the color histogram significantly, but we fixed it by improving the masks.

## 1.4 Speech Synthesis

We used the `piper-tts` library and one of the available English models to synthesize all required speech. This was mostly used for responses about ring colors and for greeting people.

# 2. Task 1r

## 2.1 Navigation

For navigation, we defined several keypoints on the map. The robot follows these keypoints in a loop, which allows it to repeat the route if it does not detect all faces during the first pass.

**TODO:** Add real-world navigation issues.

## 2.2 Face Detection

For face detection, we used the provided YOLOv8n model. Detection worked well in simulation, but we added several improvements to make detections and marker positions more robust:

- **Detection threshold:** detections from large distances are ignored because they have a higher chance of being false positives.
- **Deduplication radius:** since faces are not placed close together, detections that are only slightly offset from one another are treated as the same face.
- **Approach offset:** the robot does not drive directly into the face marker. Instead, it navigates to a position slightly in front of the marker before greeting the person.
- **Median position:** every valid detection of the same face is stored in an array. Before creating the final marker, we take the median of these positions. We chose the median instead of the mean because some detections can fall on the edge of a nearby box, which would strongly affect the mean and produce an incorrect marker position.
- **Map validation:** we added checks to verify whether a face is inside the map and whether it is on a wall.

## 2.3 Ring Detection

### 2.3.1 Detection

Ring detection was implemented using OpenCV's `HoughCircles` function, which performs Hough transform circle detection on the RGB image. We tried using the same depth-image approach as in simulation, but the depth information arrived too slowly in the real setup.

Using RGB introduced new problems, such as detecting round objects that were not rings. For example, the back of a person's head could rarely be classified as a circle. We also added checks to verify whether the ring is inside the map and whether it is near a wall or box.

**TODO:** Add the method used to reject fake rings on boxes.

### 2.3.2 Color Classification

For color classification, we captured several images of the rings from different angles. From these images, we built color histograms for each ring color.

**TODO:** Gasper: describe how the histograms are generated.

We averaged all histograms for each color. The resulting average histograms were then used for classification. When a ring was detected, we compared its histogram to the stored color histograms and incremented the counter for the most similar color in a dictionary of possible colors.

We also used Lowe's ratio test to make sure that the difference between the best and second-best matches was large enough. After a certain number of detections, the ring was assigned the color with the highest number of classifications.

We had some issues with color detection because the stick holding the ring was also sometimes included in the detection. Real-world lighting also differed from the uniform lighting in simulation, which changed the color histograms significantly. We reduced these problems by improving the masks.

## 2.4 Speech Synthesis

For speech synthesis, we publish the strings that we want the robot to say on the `rc.speak_pub` topic.

# 3. Task 2s

## 3.1 Navigation

### 3.1.1 Room 1

For navigation in the first room, we defined several keypoints on the map. The robot follows these keypoints in a loop, which allows it to repeat the route if it does not detect all faces during the first pass.

### 3.1.2 Room 2

For the second room, we needed a line-following approach. We tried several methods and eventually found that the simplest one worked best. In all approaches, we used the arm camera in the downward-facing position.

The first approach followed the line based on the bottom part of the image. We calculated the centroid and used Harris corner detection to detect crossroads. At each crossroad, the robot turned 45 degrees to the left and then continued following the line. The robot tried to keep the centroid in the middle of the camera image and adjusted its movement based on the centroid offset. The main problem with this approach was unreliable behavior at crossroads.

The second approach focused only on the centroid. We calculated the centroid and then digitally cloned it to the left by a fixed offset. The robot then tried to keep this offset centroid in the middle of the camera image. This means the robot constantly drives slightly to the left of the line and always takes the left branch at crossroads. This behaves like a depth-first search and guarantees that the whole path is searched. When the line disappears, the robot turns 180 degrees and continues once it finds the line again.

## 3.2 Face Detection

### 3.2.1 Detection

For face detection, we used the provided YOLOv8n model. Detection worked well in simulation, but we added several improvements to make detections and marker positions more robust:

- **Detection threshold:** detections from large distances are ignored because they have a higher chance of being false positives.
- **Deduplication radius:** since faces are not placed close together, detections that are only slightly offset from one another are treated as the same face.
- **Approach offset:** the robot does not drive directly into the face marker. Instead, it navigates to a position slightly in front of the marker before greeting the person.
- **Median position:** every valid detection of the same face is stored in an array. Before creating the final marker, we take the median of these positions. We chose the median instead of the mean because some detections can fall on the edge of a nearby box, which would strongly affect the mean and produce an incorrect marker position.

### 3.2.2 Classification

For person classification, we used the provided dataset. Because it was too small to train the model effectively, we augmented it with the Python `albumentations` library. For each person, we generated 50 augmented images using different intensities of perspective shifts, affine transformations, random brightness and contrast changes, motion blur, Gaussian noise, and other noise types. We then used these images to train a custom `yolo26n-cls` classification model.

For gender classification, we used the outputs of the person classification, which already included pronouns, and a separate DeepFace library. DeepFace has a built-in face analysis function that returns age, gender, and emotion.

## 3.3 Ring Detection

### 3.3.1 Detection

Ring detection was implemented using OpenCV's `HoughCircles` function, which performs Hough transform circle detection on the depth image. Using depth instead of RGB reduced the chance of misclassifying 2D ring stickers on boxes as real rings.

After detecting a circle, we used the point cloud callback to get a 3D point on the ring, offset by the detected radius. This effectively gave us the center of the ring. The remaining robustness tricks were similar to those used for face detection.

### 3.3.2 Color Classification

For color classification, we captured several images of the rings from different angles. From these images, we built color histograms for each ring color.

**TODO:** Gasper: describe how the histograms are generated.

We averaged all histograms for each color. The resulting average histograms were then used for classification. When a ring was detected, we compared its histogram to the stored color histograms and incremented the counter for the most similar color in a dictionary of possible colors.

We also used Lowe's ratio test to make sure that the difference between the best and second-best matches was large enough. After a certain number of detections, the ring was assigned the color with the highest number of classifications.

We had some issues with color detection because the stick holding the ring was also sometimes included in the detection. This changed the color histogram significantly, but we fixed it by improving the masks.

## 3.4 Speech

### 3.4.1 Speech Synthesis

We used the `piper-tts` library and one of the available English models to synthesize all required speech. In the second task, there were many more phrases to synthesize, but the method stayed the same.

### 3.4.2 Speech Recognition

We used the `speech_recognition` library to convert spoken input into text. The testing results were initially poor. The output text was often wrong but still phonetically similar, for example recognizing "barrels" as "battles". This may have been caused by our accents or by limitations of the speech recognition system.

We solved this problem with a staged search through the recognized input:

1. First, we check whether the input exactly matches one of the phrases or words in our dictionaries.
2. Then, we check whether important keywords, such as "barrels", "rings", "red anomaly", or "green anomaly", are present in the input phrase.
3. Finally, we check whether words similar to the important keywords are present in the input phrase, allowing for one or two incorrect letters.

This produced good results.

### 3.4.3 Dialogue Processing

Because dialogue processing is different for men and women, we implemented a simple state machine. Depending on the gender classification result, we use either the male or female state machine. For men, we perform the step-by-step checks described in Section 3.4.2. For women, we use conditionals and booleans to handle the required logic.

## 3.5 Cylinders

### 3.5.1 Cylinder Segmentation

For cylinder segmentation, we detect both vertical and horizontal cylinders.

The node subscribes to the OAK-D camera point cloud. For every point cloud callback, we convert the ROS `PointCloud2` message into a PCL point cloud. We then filter the points and process only those that are within 3 meters and close to the floor, because barrels are expected to be low in the image even when they are standing upright.

Next, we calculate normals for every remaining point. These normals are used together with the points during RANSAC cylinder fitting. We run RANSAC with point positions, normals, and constraints that separate vertical and horizontal barrel segmentation.

We define the expected barrel radius in advance. This allows us to filter out cylinder-like objects that are too thin, such as pipes, and also reduces noise and duplicate detections. After each detected barrel, we remove its points from the point cloud and repeat the process so that another barrel can be detected. We use the same process for horizontal barrels.

For horizontal barrels, we use the barrel axis direction and the barrel center in the camera frame to project a vector perpendicular to the axis. The point at the end of this vector acts as the robot's stop point for spill detection, which is described in the next section.

### 3.5.2 Cylinder Spills

For spill detection, we also use the OAK-D point cloud. The robot first moves to the perpendicular direction of the barrel, stops for 3 seconds, and calls the `/detect_spill` service. During this 3-second window, the service processes incoming point cloud frames and detects possible spills.

The process first removes all `inf` and `NaN` values. It then removes points near the horizontal barrel positions, which are known from cylinder segmentation. After that, it keeps only points with heights between `0.02 m` and `0.1 m`.

The remaining points are discretized into `0.05 m` grid cells, and connected components are run on this grid. This gives us the number of clusters and the points belonging to each cluster. We then iterate through the clusters and run a final checker function that verifies whether the cluster has the expected spill shape.

This is repeated for every point cloud callback during the 3-second window. If more than 50% of the processed frames are classified as containing a spill, the horizontal barrel is classified as spilled.

## 3.6 PDF Generation

For PDF generation, we use the `reportlab` library. All required information is already stored in `robot_commander`, so this part only creates the final report. An example of this generated report is included with this report.

## 3.7 Anomaly Detection

For anomaly detection, we use a custom-trained SuperSimpleNet. We found a checkpoint on Hugging Face that had been trained on a similar tile dataset and then continued training it on our own dataset.

The anomaly detection itself was the simpler part. The more difficult part was getting the robot to and from the conveyor belt, and then extracting only the tile area for inference.

For movement toward the conveyor belt, we used the top camera looking downward. From a fixed waypoint, the robot slowly moves forward while running Otsu thresholding. If Otsu's between-class variance reaches a certain threshold, we assume that the robot is at the conveyor. The robot then turns right to confirm that the conveyor color matches the requested color. After confirmation, it moves forward with the camera positioned to the left of the robot.

While driving forward, we use Hough lines on the tile image to get four tile edge lines. The intersections of these lines are used as the tile corners. We then use a homography to transform the tile into a clean square image, which is used for anomaly detection.

To keep the robot centered on the tiles and to count which tile is currently being inspected, we use a non-Hough tile detection method. We convert the camera image to grayscale, blur it to reduce noise, and binarize it. We then use closing and opening morphology operations to remove additional imperfections.

Next, we run connected components to separate each white region and assign it a label. We pick the largest region, assuming it is the tile, and create a binary mask for it. We then find the contour of that tile using `findContours` and extract four corner points.

These four points are used to calculate the horizontal offset, vertical offset, and angle of the tile. Based on these values, we adjust the robot's movement if it is misaligned.

For counting, we check whether the center of the image is covered by the area inside the tile's four corners. Once the tile has covered the center and then leaves it, we increment the tile counter.

## 3.8 Gender Classification

For gender classification, we use the DeepFace library and its `analyze` function. It can return predicted age, emotion, gender, and other attributes. We use it to get the predicted gender and the confidence or probability of that prediction.
