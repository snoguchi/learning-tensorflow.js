const tf = require('@tensorflow/tfjs');
require('@tensorflow/tfjs-node');

// https://js.tensorflow.org/tutorials/core-concepts.html

// Tensors
{
  const shape = [2, 3];
  const a = tf.tensor([1.0, 2.0, 3.0, 10.0, 20.0, 30.0], shape);
  a.print();

  const b = tf.tensor([[1.0, 2.0, 3.0], [10.0, 20.0, 30.0]]);
  b.print();

  const c = tf.tensor2d([[1.0, 2.0, 3.0], [10.0, 20.0, 30.0]]);
  c.print();

  const zeros = tf.zeros([3, 5]);
  zeros.print();
}


// Variables
{
  const initialValues = tf.zeros([5]);
  const biases = tf.variable(initialValues);
  biases.print();

  const updatedValues = tf.tensor1d([0, 1, 0, 1, 0]);
  biases.assign(updatedValues);
  biases.print();
}


// Operations (Ops)
{
  const d = tf.tensor2d([[1.0, 2.0], [3.0, 4.0]]);
  const d_squared = d.square();
  d_squared.print();

  const e = tf.tensor2d([[1.0, 2.0], [3.0, 4.0]]);
  const f = tf.tensor2d([[5.0, 6.0], [7.0, 8.0]]);
  const e_plus_f = e.add(f);
  e_plus_f.print();

  const sq_sum = e.add(f).square();
  sq_sum.print();
}


// Models and Layers
{
  function predict(input) {
    // y = a * x ^ 2 + b * x + c
    return tf.tidy(() => {
      const x = tf.scalar(input);
      const ax2 = a.mul(x.square());
      const bx = b.mul(x);
      const y = ax2.add(bx).add(c);
      return y;
    });
  }

  const a = tf.scalar(2);
  const b = tf.scalar(4);
  const c = tf.scalar(8);

  const result = predict(2);
  result.print();
}

if (0) {
  const model = tf.sequential();
  model.add(
    tf.layers.simpleRNN({
      units: 20,
      recurrentInitializer: 'GlorotNormal',
      inputShape: [80, 4]
    })
  );

  const optimizer = tf.train.sgd(LEARNING_RATE);
  model.compile({optimizer, loss: 'categoricalCrossentropy'});
  model.fit({x: data, y: label});
}


// Memory Management: dispose and tf.tidy
{
  const x = tf.tensor2d([[0.0, 2.0], [4.0, 6.0]]);
  const x_squared = x.square();

  x.dispose();
  x_squared.dispose();
}

{
  const average = tf.tidy(() => {
    const y = tf.tensor1d([1.0, 2.0, 3.0, 4.0]);
    const z = tf.ones([4]);
    return y.sub(z).square().mean();
  });

  average.print();
}
