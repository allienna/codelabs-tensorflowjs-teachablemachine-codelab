# [codelab] TensorFlow.js Transfer Learning Image Classifier

> Information can be found [here](https://codelabs.developers.google.com/codelabs/tensorflowjs-teachablemachine-codelab).
>
> While Google could remove instructions, I copy/paste them below.

## About this codelab

Written by Nikhil Thorat

## 1. Introduction

In this codelab, you will learn how to build a simple "[teachable machine](https://teachablemachine.withgoogle.com/)", a
custom image classifier that you will train on the fly in the browser using TensorFlow.js, a powerful and flexible
machine learning library for Javascript. You will first load and run a popular pre-trained model called MobileNet for
image classification in the browser. You will then use a technique called "transfer learning", which bootstraps our
training with the pre-trained MobileNet model and customizes it to train for your application.

This codelab will not go over the theory behind the teachable machine application. If you are curious about that, check
out [this tutorial](https://beta.observablehq.com/@nsthorat/how-to-build-a-teachable-machine-with-tensorflow-js).

### What you'll learn

- How to load pretrained MobileNet model and make a prediction on new data
- How to make predictions through the webcam
- How to use intermediate activations of MobileNet to do transfer learning on a new set of classes you define on the fly
  with the webcam

So let's get started!

## 2. Requirements

To complete this codelab, you will need:

1. A recent version of Chrome or another modern browser.
2. A text editor, either running locally on your machine or on the web via something like Codepen or Glitch.
3. Knowledge of HTML, CSS, JavaScript, and Chrome DevTools (or your preferred browsers devtools).
4. A high-level conceptual understanding of Neural Networks. If you need an introduction or refresher, consider watching
   this [video by 3blue1brown](https://www.youtube.com/watch?v=aircAruvnKk) or
   this [video on Deep Learning in Javascript by Ashi Krishnan](https://www.youtube.com/watch?v=SV-cgdobtTA).

> Note: If you are at a CodeLab kiosk we recommend using glitch.com to complete this codelab. We have set up a [starter
> project for you to remix](https://glitch.com/~tfjs-glitch-starter) that loads tensorflow.js.

## 3. Load TensorFlow.js and MobileNet Model

Open **index.html** in an editor and add this content:

```html

<html>
<head>
    <!-- Load the latest version of TensorFlow.js -->
    <script src="https://cdn.jsdelivr.net/npm/@tensorflow/tfjs"></script>
    <script src="https://cdn.jsdelivr.net/npm/@tensorflow-models/mobilenet"></script>
</head>
<body>
<div id="console"></div>
<!-- Add an image that we will use to test -->
<img id="img" crossorigin src="https://i.imgur.com/JlUvsxa.jpg" width="227" height="227"/>
<!-- Load index.js after the content of the page -->
<script src="index.js"></script>
</body>
</html>
```

## 4. Set up MobileNet for inference in browser

Next, Open/Create file index.js in a code editor, and include the following code:

```javascript
let net;

async function app() {
    console.log('Loading mobilenet..');

    // Load the model.
    net = await mobilenet.load();
    console.log('Successfully loaded model');

    // Make a prediction through the model on our image.
    const imgEl = document.getElementById('img');
    const result = await net.classify(imgEl);
    console.log(result);
}

app();
```

## 5. Test MobileNet Inference in Browser

To run the webpage, simply open **index.html** in a Web Browser. If you are using the cloud console, simply refresh the
preview page.

You should see a picture of a dog and in the Javascript console in Developer Tools, the top predictions of MobileNet!
Note that this may take a little bit of time to download the model, be patient!

Did the image get classified correctly?

It's also worth noting that this will also work on a mobile phone!

## 6. Run MobileNet Inference in Browser through Webcam images

Now, let's make this more interactive and real-time. Let's set up the webcam to make predictions on images that come
through the webcam.

First set up the webcam video element. Open the **index.html** file, and add the following line inside the <body>
section and delete the <img> tag we had for loading the dog image:

```html

<video autoplay playsinline muted id="webcam" width="224" height="224"></video>
```

Open the **index.js** file and add the webcamElement to the very top of the file

```javascript
const webcamElement = document.getElementById('webcam');
```

Now, in the **app()** function which you added before, you can remove the prediction through the image and instead
create an infinite loop which makes predictions through the webcam element.

```javascript
async function app() {
    console.log('Loading mobilenet..');

// Load the model.
    net = await mobilenet.load();
    console.log('Successfully loaded model');

// Create an object from Tensorflow.js data API which could capture image
// from the web camera as Tensor.
    const webcam = await tf.data.webcam(webcamElement);
    while (true) {
        const img = await webcam.capture();
        const result = await net.classify(img);

        document.getElementById('console').innerText = `
      prediction: ${result[0].className}\n
      probability: ${result[0].probability}
    `;
        // Dispose the tensor to release the memory.
        img.dispose();

        // Give some breathing room by waiting for the next animation frame to
        // fire.
        await tf.nextFrame();
    }
}
```

If you open your console in the webpage, you should now see a MobileNet prediction with probability for every frame
collected on the webcam.

These may be nonsensical because the ImageNet dataset does not look very much like images that would typically appear in
a webcam. One way to test this is by holding a picture of a dog on your phone in front of your laptop camera.

## 7. Add a custom classifier on top of the MobileNet predictions

Now, let's make this more useful. We will make a custom 3-class object classifier using the webcam on the fly. We're
going to make a classification through MobileNet, but this time we will take an internal representation (activation) of
the model for a particular webcam image and use that for classification.

We'll use a module called a "K-Nearest Neighbors Classifier", which effectively lets us put webcam images (actually,
their MobileNet activations) into different categories (or "classes"), and when the user asks to make a prediction we
simply choose the class that has the most similar activation to the one we are making a prediction for.

Add an import of KNN Classifier to the end of the imports in the <head> tag of **index.html** (you will still need
MobileNet, so don't remove that import):

```html
...
<script src="https://cdn.jsdelivr.net/npm/@tensorflow-models/knn-classifier"></script>
...
```

Add 3 buttons for each of the buttons in **index.html** below the video element. These buttons will be used to add
training images to the model.

```html
...
<button id="class-a">Add A</button>
<button id="class-b">Add B</button>
<button id="class-c">Add C</button>
...
```

At the top of **index.js**, create the classifier:

```javascript
const classifier = knnClassifier.create();
```

Update the app function:

```javascript
async function app() {
    console.log('Loading mobilenet..');

    // Load the model.
    net = await mobilenet.load();
    console.log('Successfully loaded model');

    // Create an object from Tensorflow.js data API which could capture image
    // from the web camera as Tensor.
    const webcam = await tf.data.webcam(webcamElement);

    // Reads an image from the webcam and associates it with a specific class
    // index.
    const addExample = async classId => {
        // Capture an image from the web camera.
        const img = await webcam.capture();

        // Get the intermediate activation of MobileNet 'conv_preds' and pass that
        // to the KNN classifier.
        const activation = net.infer(img, true);

        // Pass the intermediate activation to the classifier.
        classifier.addExample(activation, classId);

        // Dispose the tensor to release the memory.
        img.dispose();
    };

    // When clicking a button, add an example for that class.
    document.getElementById('class-a').addEventListener('click', () => addExample(0));
    document.getElementById('class-b').addEventListener('click', () => addExample(1));
    document.getElementById('class-c').addEventListener('click', () => addExample(2));

    while (true) {
        if (classifier.getNumClasses() > 0) {
            const img = await webcam.capture();

            // Get the activation from mobilenet from the webcam.
            const activation = net.infer(img, 'conv_preds');
            // Get the most likely class and confidence from the classifier module.
            const result = await classifier.predictClass(activation);

            const classes = ['A', 'B', 'C'];
            document.getElementById('console').innerText = `
        prediction: ${classes[result.label]}\n
        probability: ${result.confidences[result.label]}
      `;

            // Dispose the tensor to release the memory.
            img.dispose();
        }

        await tf.nextFrame();
    }
}
```

Now when you load the index.html page, you can use common objects or face/body gestures to capture images for each of
the three classes. Each time you click one of the "Add" buttons, one image is added to that class as a training example.
While you do this, the model continues to make predictions on webcam images coming in and shows the results in
real-time.

## 8. Optional: extending the example

Try now adding another class that represents no action!

## 9. What you learned

In this codelab, you implemented a simple machine learning web application using TensorFlow.js. You loaded and used a
pretrained MobileNet model for classifying images from webcam. You then customized the model to classify images into
three custom categories.

Be sure to visit [js.tensorflow.org](https://js.tensorflow.org/) for more examples and demos with code to see how you
can use TensorFlow.js in your applications.