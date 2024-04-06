import * as tf from "@tensorflow/tfjs-node";
import data from "./pokemon-api-data.json" assert { type: "json" };
const cards = data.map((card, index) => ({
  label: index,
  imageUrl: card.imageUrl,
}));

// Function to fetch an image from a URL and convert it into a Tensor
async function preprocessImageFromUrl(imageUrl: any) {
  const response = await fetch(imageUrl);
  const arrayBuffer = await response.arrayBuffer(); // Get the image as an ArrayBuffer
  const buffer = Buffer.from(arrayBuffer); // Convert ArrayBuffer to Node.js Buffer
  let tensor = tf.node.decodeImage(buffer, 3);
  tensor = tf.image.resizeBilinear(tensor, [825, 600]); // Adjusted to match the model input
  tensor = tensor.div(tf.scalar(255.0)).sub(tf.scalar(0.5)).mul(tf.scalar(2)); // Normalize
  return tensor.expandDims(); // Add batch dimension
}

// Data generator function
function createDatasetGenerator() {
    return async function* dataGenerator(): any {
    for (let i = 0; i < cards.length; i += 32) {
        const batchCards = cards.slice(i, i + 32);
        const xs: tf.Tensor<tf.Rank>[] = [];
        const ys: number[] = [];

        for (const card of batchCards) {
        const imageTensor = await preprocessImageFromUrl(card.imageUrl);
        xs.push(imageTensor.squeeze());
        // Assuming labels are numerical ids for simplicity
        ys.push(card.label);
        }

        const labelsTensor = tf.oneHot(
        tf.tensor1d(ys, "int32"),
        cards.length /* number of classes */
        );
        yield { xs: tf.stack(xs), ys: labelsTensor };
    }
    }
}

// Define the model
function createModel(numClasses: number) {
  const model = tf.sequential();
  model.add(
    tf.layers.conv2d({
      inputShape: [825, 600, 3],
      filters: 32,
      kernelSize: 3,
      activation: "relu",
    })
  );
  model.add(tf.layers.maxPooling2d({ poolSize: 2 }));
  model.add(tf.layers.flatten());
  model.add(tf.layers.dense({ units: 128, activation: "relu" }));
  model.add(tf.layers.dense({ units: numClasses, activation: "softmax" }));
  return model;
}

async function trainModel() {
  const numClasses = new Set(cards.map((card) => card.label)).size;
  const model = createModel(numClasses);
  model.compile({
    optimizer: "adam",
    loss: "categoricalCrossentropy",
    metrics: ["accuracy"],
  });

  const batchSize = 32;
  await model.fitDataset(tf.data.generator(createDatasetGenerator()), {
    epochs: 10,
    batchesPerEpoch: Math.ceil(cards.length / batchSize),
  });

  await model.save("file://./pokemon-tcg-detector-model");
}

trainModel().then(() => console.log("Model trained successfully"));
