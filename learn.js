// code here
import * as tf from '@tensorflow/tfjs';
//import labels from './imagenet_labels.json';

import blue1 from './data/colors/training/blue/blue-1.png';
import blue2 from './data/colors/training/blue/blue-2.png';
import blue3 from './data/colors/validation/blue/blue-3.png';
import red1 from './data/colors/training/red/red-1.png';
import red2 from './data/colors/training/red/red-2.png';
import red3 from './data/colors/validation/red/red-3.png';

const training = [
  blue1,
  blue2,
  red1,
  red2,
];

// labels should match the positions of their associated images
const labels = [
  'blue',
  'blue',
  'red',
  'red',
];


buildPretrainedModel().then(pretrainedModel => {
   
    loadImages(training, pretrainedModel).then(xs => {
      const ys = addLabels(labels);
  
      const model = getModel(2);
  
      model.fit(xs, ys, {
        epochs: 20,
        shuffle: true,
      }).then(history => {
        // make predictions
        makePrediction(pretrainedModel, blue3, "0", model);
        makePrediction(pretrainedModel, red3, "1", model);
      });
    });
  });




  
  function makePrediction(pretrainedModel, image, expectedLabel, model) {
    loadImage(image).then(loadedImage => {
      return loadAndProcessImage(loadedImage);
    }).then(loadedImage => {
      const activatedImage = pretrainedModel.predict(loadedImage);
      loadedImage.dispose();
      return activatedImage;
    }).then(activatedImage => {
      const prediction = model.predict(activatedImage);
      const predictionLabel = prediction.as1D().argMax().dataSync()[0];
  
      console.log('Expected Label', expectedLabel);
      console.log('Predicted Label', predictionLabel);
  
      prediction.dispose();
      activatedImage.dispose();
    });
  }




  function buildPretrainedModel() {
    return loadMobilenet().then(mobilenet => {
      const layer = mobilenet.getLayer('conv_pw_13_relu');
      return tf.model({
        inputs: mobilenet.inputs,
        outputs: layer.output,
      });
    });
  }


  function loadImages(images, pretrainedModel) {
    let promise = Promise.resolve();
    for (let i = 0; i < images.length; i++) {
      const image = images[i];
      promise = promise.then(data => {
        return loadImage(image).then(loadedImage => {
          // Note the use of `tf.tidy` and `.dispose()`. These are two memory management
          // functions that Tensorflow.js exposes.
          // https://js.tensorflow.org/tutorials/core-concepts.html
          //
          // Handling memory management is crucial for building a performant machine learning
          // model in a browser.
          return tf.tidy(() => {
            const processedImage = loadAndProcessImage(loadedImage);
            const prediction = pretrainedModel.predict(processedImage);
  
            if (data) {
              const newData = data.concat(prediction);
              data.dispose();
              return newData;
            }
  
            return tf.keep(prediction);
          });
        });
      });
    }
  
    return promise;
  }





/*
loadMobilenet().then(pretrainedModel => {
  loadImage(drum).then(img => {
    const processedImage = loadAndProcessImage(img);
    const prediction = pretrainedModel.predict(processedImage);

    // Because of the way Tensorflow.js works, you must call print on a Tensor instead of console.log.
    //prediction.print();
     const labelPrediction = prediction.as1D().argMax().dataSync()[0];
     console.log(`
     Numeric prediction is ${labelPrediction}
     The predicted label is ${labels[labelPrediction]}
     The actual label is drum, membranophone, tympan
   `);
  });
});
*/

function oneHot(labelIndex, classLength) {
    return tf.tidy(() => tf.oneHot(tf.tensor1d([labelIndex]).toInt(), classLength));
  };

  function getLabelsAsObject(labels) {
    let labelObject = {};
    for (let i = 0; i < labels.length; i++) {
      const label = labels[i];
      if (labelObject[label] === undefined) {
        // only assign it if we haven't seen it before
        labelObject[label] = Object.keys(labelObject).length;
      }
    }
    return labelObject;
  }

  function addLabels(labels) {
    return tf.tidy(() => {
      const classes = getLabelsAsObject(labels);
      const classLength = Object.keys(classes).length;
  
      let ys;
      for (let i = 0; i < labels.length; i++) {
        const label = labels[i];
        const labelIndex = classes[label];
        const y = oneHot(labelIndex, classLength);
        if (i === 0) {
          ys = y;
        } else {
          ys = ys.concat(y, 0);
        }
      }
      return ys;
    });
  };



  function getModel(numberOfClasses) {
    const model = tf.sequential({
      layers: [
        tf.layers.flatten({inputShape: [7, 7, 256]}),
        tf.layers.dense({
          units: 100,
          activation: 'relu',
          kernelInitializer: 'varianceScaling',
          useBias: true
        }),
        tf.layers.dense({
          units: numberOfClasses,
          kernelInitializer: 'varianceScaling',
          useBias: false,
          activation: 'softmax'
        })
      ],
    });
  
    model.compile({
      optimizer: tf.train.adam(0.0001),
      loss: 'categoricalCrossentropy',
      metrics: ['accuracy'],
    });
  
    return model;
  }



function loadMobilenet() {
    //return tf.loadModel('./models/mobilenet/model.json');
    return tf.loadModel('https://storage.googleapis.com/tfjs-models/tfjs/mobilenet_v1_0.25_224/model.json');
  }

  function loadImage(src) {
    return new Promise((resolve, reject) => {
      const img = new Image();
      img.src = src;
      img.onload = () => resolve(tf.fromPixels(img));
      img.onerror = (err) => reject(err);
    });
  }

  function cropImage(img) {
    const width = img.shape[0];
    const height = img.shape[1];
  
    // use the shorter side as the size to which we will crop
    const shorterSide = Math.min(img.shape[0], img.shape[1]);
  
    // calculate beginning and ending crop points
    const startingHeight = (height - shorterSide) / 2;
    const startingWidth = (width - shorterSide) / 2;
    const endingHeight = startingHeight + shorterSide;
    const endingWidth = startingWidth + shorterSide;
  
    // return image data cropped to those points
    return img.slice([startingWidth, startingHeight, 0], [endingWidth, endingHeight, 3]);
  }


  function resizeImage(image) {
    return tf.image.resizeBilinear(image, [224, 224]);
  }


  function batchImage(image) {
    // Expand our tensor to have an additional dimension, whose size is 1
    const batchedImage = image.expandDims(0);
  
    // Turn pixel data into a float between -1 and 1.
    return batchedImage.toFloat().div(tf.scalar(127)).sub(tf.scalar(1));
  }


  function loadAndProcessImage(image) {
    const croppedImage = cropImage(image);
    const resizedImage = resizeImage(croppedImage);
    const batchedImage = batchImage(resizedImage);
    return batchedImage;
  }