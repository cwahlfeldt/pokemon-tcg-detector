import * as tf from '@tensorflow/tfjs-node';
import data from './pokemon-api-data.json' assert { type: "json" };
const cards = data.map((card) => ({ label: card.id, imageUrl: card.imageUrl }));

// Function to fetch an image from a URL and convert it into a Tensor
async function preprocessImageFromUrl(imageUrl: any) {
    const response = await fetch(imageUrl);
    const arrayBuffer = await response.arrayBuffer(); // Get the image as an ArrayBuffer
    const buffer = Buffer.from(arrayBuffer); // Convert ArrayBuffer to Node.js Buffer

    // Decode the image to a Tensor
    let tensor = tf.node.decodeImage(buffer, 3);

    // Resize the image to a specific size while keeping the aspect ratio the same
    // Adjust the target width and height according to your model's input size
    const targetWidth = 600;
    const targetHeight = 825;
    tensor = tf.image.resizeBilinear(tensor, [targetHeight, targetWidth]);

    // Normalize the image data to a range between -1 and 1 (this is model-dependent)
    tensor = tensor.toFloat().div(tf.scalar(127.5)).sub(tf.scalar(1));

    // Add an additional dimension to match the input shape of the model: [1, height, width, 3]
    tensor = tensor.expandDims(0);

    return tensor;
}

async function loadDataset() {
  const images = [];
  const labels = [];
  const labelIndex: any = {}; // To convert string labels to integers

  for (let {imageUrl, label} of cards) {
      const tensor = await preprocessImageFromUrl(imageUrl);
      images.push(tensor);

      // Convert label to integer index
      if (!(label in labelIndex)) {
          labelIndex[label] = Object.keys(labelIndex).length;
      }
      labels.push(labelIndex[label]);
  }

  // Convert lists of tensors into a single tensor
  const imagesTensor = tf.concat(images);
  const labelsTensor = tf.oneHot(tf.tensor1d(labels, 'int32'), Object.keys(labelIndex).length);

  return {imagesTensor, labelsTensor, labelIndex};
}

// Example usage
// const imageUrl = 'YOUR_IMAGE_URL_HERE';
// preprocessImageFromUrl(imageUrl).then(tensor => {
//     console.log('Preprocessed image tensor:', tensor);
//     // Here, you could pass the tensor to your model for prediction
// });


// Define a simple model for classification
function createModel() {
    const model = tf.sequential();
    model.add(tf.layers.conv2d({inputShape: [200, 200, 3], filters: 32, kernelSize: 3, activation: 'relu'}));
    model.add(tf.layers.maxPooling2d({poolSize: 2}));
    model.add(tf.layers.flatten());
    model.add(tf.layers.dense({units: 128, activation: 'relu'}));
    model.add(tf.layers.dense({units: cards.length, activation: 'softmax'}));
    return model;
}

async function trainModel() {
    const model = createModel();
    model.compile({
        optimizer: 'adam',
        loss: 'categoricalCrossentropy',
        metrics: ['accuracy'],
    });

    const {trainX, trainY}: any = await loadDataset();
    
    await model.fit(trainX, trainY, {
        epochs: 10,
        batchSize: 32,
    });

    await model.save('file://./my-pokemon-model');
}

trainModel().then(() => console.log('Model trained successfully'));
