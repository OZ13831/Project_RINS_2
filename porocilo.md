# 1. Task1s
## 1.1 Navigation

For navigation we set several keypoints on the map, where the robot walked around which would loop in case that our robot didn't detect all the faces in the first iteration of the run.

## 1.2 Face Detection

For face detection we use the yolov8n model that we already got given. The detection in the simulation worked well, but we needed to add some adjustments so that the detections and then positions of the markers were more robust. The biggest things were: 
- Detect threshold, which made sure that detections at a distance (which had a higher chance of being a false detection) were ignored.
- Deduplication radius, because we knew that faces won't be set close together, this made sure that slightly offset detections of the same face were not seen as different faces.
- Approach offset, this made sure that our robot didn't try to go directly into the marker of the robot but instead went to the position slightly away from the marker before greeting.
- Median of positions, whenever we saw a face, (which wasn't too far) we put every single detection of said face into one array, before making the actual marker, we looked at those positions and took the median as the position that would eventually be the marker. We chose the median instead of the mean, because the robot would sometimes detect the center points of the face on the edge of a nearby box, which would greatly effect the mean calculation and at the end make the result wrong.


## 1.3 Ring Detection
### 1.3.1 Detection

The actual detection part, we did by using the opencv library houghCircles function, which does hough transform circle detection on the depth image. Using depth instead of RGB removed the chance of us missclassifying the 2D ring stickers that were put on boxes. We used the information about the detected circles to then return a point from the pointcloud callback on the ring offset by the radius, which effectively gave us the center of the ring. The rest of the tricks for robustness were the same as for face detection.

### 1.3.2 Color classification

For color classification we made several images of the rings, from different angles. We built a model that made that those histograms by :

GASPER OPISI KAKO DELUJE IZDLEOVANJE HISTOGRAMOV

we then averaged all of the made histograms. The resulting histograms were used to then classify color. When we got a detection of the ring, we looked at which histogram the ring is most similar to and then incremented the counter inside a dictionary of colors we knew were possible. We also used IOWE's ration to make sure that the difference between the best and second best similarity was high enough. After a certain number of detections, we atributed the color of the ring to the color with the highest number of classifications in the dictionary of colors. We had some issues with color detection, because the stick that the ring was on was also being detected, this made the color histogram of the image completly change but we managed to fix it by making better masks.

## 1.4 Voice Synthetization

We used the piper-tts library and one of the english models that were available to sythesise anything needed. This were mostly he responses to colors of the rings and greets to people.

# 2. Task1r
## 2.1 Navigation

For navigation we set several keypoints on the map, where the robot walked around which would loop in case that our robot didn't detect all the faces in the first iteration of the run. DOPISI TEZAVE

## 2.2 Face Detection

For face detection we use the yolov8n model that we already got given. The detection in the simulation worked well, but we needed to add some adjustments so that the detections and then positions of the markers were more robust. The biggest things were: 
- Detect threshold, which made sure that detections at a distance (which had a higher chance of being a false detection) were ignored.
- Deduplication radius, because we knew that faces won't be set close together, this made sure that slightly offset detections of the same face were not seen as different faces.
- Approach offset, this made sure that our robot didn't try to go directly into the marker of the robot but instead went to the position slightly away from the marker before greeting.
- Median of positions, whenever we saw a face, (which wasn't too far) we put every single detection of said face into one array, before making the actual marker, we looked at those positions and took the median as the position that would eventually be the marker. We chose the median instead of the mean, because the robot would sometimes detect the center points of the face on the edge of a nearby box, which would greatly effect the mean calculation and at the end make the result wrong.
- We also added checkers if the face is inside the map and if so, if its on a wall.

## 2.3 Ring Detection
### 2.3.1 Detection

The actual detection part, we did by using the opencv library houghCircles function, which does Hough transform circle detection on the RGB image. We tried using the same depth image Hough transform as in the simulation but the depth image information comes too slow. This introduced new problems like detecing any round objects instead of just rings. Rarely but sometimes a round back of the head gets classified as a circle. DOPISI how tf smo nrdil da ne zazna unih fake krogov na skatlah. We also added checkers if the ring is inside the map and if so, if its near a wall (box).

### 2.3.2 Color classification

For color classification we made several images of the rings, from different angles. We built a model that made that those histograms by :

GASPER OPISI KAKO DELUJE IZDLEOVANJE HISTOGRAMOV

we then averaged all of the made histograms. The resulting histograms were used to then classify color. When we got a detection of the ring, we looked at which histogram the ring is most similar to and then incremented the counter inside a dictionary of colors we knew were possible. We also used IOWE's ration to make sure that the difference between the best and second best similarity was high enough. After a certain number of detections, we atributed the color of the ring to the color with the highest number of classifications in the dictionary of colors. We had some issues with color detection, because the stick that the ring was on was also being detected and because of lighting in real life compared to the uniform lighting in the simulation, this made the color histogram of the image completly change but we managed to fix it by making better masks.

## 2.4 Voice Synthetization



# 3. Task2s
## 3.1 Navigation
### 3.1.1 Room 1

For navigation we set several keypoints on the map, where the robot walked around which would loop in case that our robot didn't detect all the faces in the first iteration of the run.

### 3.1.2 Room 2



## 3.2 Face Detection
### 3.2.1 Detection

For face detection we use the yolov8n model that we already got given. The detection in the simulation worked well, but we needed to add some adjustments so that the detections and then positions of the markers were more robust. The biggest things were: 
- Detect threshold, which made sure that detections at a distance (which had a higher chance of being a false detection) were ignored.
- Deduplication radius, because we knew that faces won't be set close together, this made sure that slightly offset detections of the same face were not seen as different faces.
- Approach offset, this made sure that our robot didn't try to go directly into the marker of the robot but instead went to the position slightly away from the marker before greeting.
- Median of positions, whenever we saw a face, (which wasn't too far) we put every single detection of said face into one array, before making the actual marker, we looked at those positions and took the median as the position that would eventually be the marker. We chose the median instead of the mean, because the robot would sometimes detect the center points of the face on the edge of a nearby box, which would greatly effect the mean calculation and at the end make the result wrong.

### 3.2.2 Classification

For person classification we used the given dataset. Because it was to small to effectively train the model, we augmented it using the python albumentations library. For each person we made 50 images that are augmented using different intesity of perspective shift, affine transformations, random brightness and contrast changes, motion blur, gaussian noise and other noise. We then used these to train a custom yolo26n-cls model for classification.

For gender classification we used the outputs of the outputs of the classification which had prounouns in there already and a seperate DeepFace library which had a builtin face analyzation function which returned us age, gender and even emotion.


## 3.3 Ring Detection
### 3.3.1 Detection

The actual detection part, we did by using the opencv library houghCircles function, which does hough transform circle detection on the depth image. Using depth instead of RGB removed the chance of us missclassifying the 2D ring stickers that were put on boxes. We used the information about the detected circles to then return a point from the pointcloud callback on the ring offset by the radius, which effectively gave us the center of the ring. The rest of the tricks for robustness were the same as for face detection.

### 3.3.2 Color classification

For color classification we made several images of the rings, from different angles. We built a model that made that those histograms by :

GASPER OPISI KAKO DELUJE IZDLEOVANJE HISTOGRAMOV

we then averaged all of the made histograms. The resulting histograms were used to then classify color. When we got a detection of the ring, we looked at which histogram the ring is most similar to and then incremented the counter inside a dictionary of colors we knew were possible. We also used IOWE's ration to make sure that the difference between the best and second best similarity was high enough. After a certain number of detections, we atributed the color of the ring to the color with the highest number of classifications in the dictionary of colors. We had some issues with color detection, because the stick that the ring was on was also being detected, this made the color histogram of the image completly change but we managed to fix it by making better masks.

## 3.4 Speech 
### 3.4.1 Speech Synthetization

We used the piper-tts library and one of the english models that were available to sythesise anything needed. In the second task there were a lot more phrases that we synthesized but the way we did it stayed the same.

### 3.4.2 Speech Recognition

We used the speech recognition library to convert inputs from speech to text. The testing results were pretty poor and the output text was ofter wrong even tho very similar, making mistakes like barrels -> battles, this could have been because of the accents we have or because of bad speech recognition. We solved this problem by doing a staged search through the inputs. 

-We first look if the input is exactly the same as some of the phrases/words we have in our dictionaries. 
-Second we check if there are important keywords, like barrels, rings, red anomaly, green anomly present in the input phrase. 
-Lastly we check if there are words very similar (one or two letters off) to the important keywords present in the input phrase.

This gave us good results.

### 3.4.1 Dialogue processing

Because the dialogue processing is different for men and women, we made a simple state machine. Depending on who we are talking to, we will know this from gender classification we either use the male of female state machine. For men we just do the step by step check mentioned in 3.4.2, while for women we have if statements and booleans in place to take care of the correct logic.

## 3.5 Cylinder detection


## 3.6 PDF-generation


## 3.7 Anomaly detection
