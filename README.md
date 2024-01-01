# turbo-annotator
Personal project for helping me to annotate images for object detection and object classification.

# Features

* This app let you load a folder with the images you will use to find objects.
* It uses SAM for suggesting the objects to be found.
* It save the results in JSON with all the information needed for saving the detected objects.

# Some notes

* The most time consuming process in SAM when using automatic mask generator is the evaluation of several prompts. 
    * For automaks it takes 7.5s for an image.
    * Using a prompt it takes 1.7s for encoding and 0.04s for decoding a prompt.
