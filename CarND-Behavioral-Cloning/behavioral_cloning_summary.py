"""
drive.py: a Python script that you can use to drive the car autonomously, once your deep neural network model is trained
video.py: a script that can be used to make a video of the vehicle when it is driving autonomously
writeup_template.md: a writeup template

sample driving data for the first track: /opt/carnd_p3/data/ (/opt is in the directory above /home, where your workspace is contained) when using GPU mode only
->you may need to collect additional data in order to get the vehicle to stay on the road.
->You can toggle record by pressing R, previously you had to click the record button
->You can takeover in autonomous mode. While W or S are held down you can control the car the same way you would in training mode.
->We suggest running at the smallest size and the fastest graphical quality.

in the folder to download the zip:
wget https://www.dropbox.com/sh/ffqzzi18b0n4r3b/AAB-f-MbQLPcdZTrcpy9X5T8a?dl=0
unzip AAB-f-MbQLPcdZTrcpy9X5T8a?dl=0

Collect more data
--------------------------------------------------------
We need to teach the car what to do when it’s off on the side of the road.
    One approach might be to constantly wander off to the side of the road
    and then steer back to the middle. A better approach is to only record data
    when the car is driving from the side of the road back toward the center line.

Driving Counter-Clockwise
    Track one has a left turn bias. If you only drive around the first track
    in a clock-wise direction, the data will be biased towards left turns.

Using Both Tracks
    If you end up using data from only track one, the convolutional neural network
    could essentially memorize the track.

Here are some general guidelines for data collection:
    two or three laps of center lane driving
    one lap of recovery driving from the sides
    one lap focusing on driving smoothly around curve

Data Generation
--------------------------------------------------------
*driving counter-clockwise can help the model generalize
*flipping the images is a quick way to augment the data
*collecting data from the second track can also help generalize the model

*the car should stay in the center of the road as much as possible
*if the car veers off to the side, it should recover back to center

*we want to avoid overfitting or underfitting when training the model
*knowing when to stop collecting more data

Data Augmentation
--------------------------------------------------------
flip the images and take the opposite sign of the steering measurement:
    image_flipped = np.fliplr(image)
    measurement_flipped = -measurement
Other options:
    change brightness
    shift horizontally or vertically

Side images serve to teach the network to steer more in a direction when they appear:
    when an image like this appears take the steering measurement and
        add a constant (say 0.2) from the steering for left images
        subtract a (same) constant to steering for right images
    Loop through the first 3 tokens of each line in the csv line, and use these
    to load the images to the images array. Then we add 3 measurements to the measurements array
    corresponding to the added images.

    During training, you want to feed the left and right camera images to your model
    as if they were coming from the center camera. This way,
    you can teach your model how to steer if the car drifts off to the left or the right.

    During prediction (i.e. "autonomous mode"), you only need to predict
    with the center camera image.
    It is not necessary to use the left and right images to derive a successful model.
    Recording recovery driving from the sides of the road is also effective.

"""

