use crate::{
    layer::Layer,
    math::{Matrix, Scalar, Vector},
};

pub struct Dense<T: Scalar, const I: usize, const N: usize> {
    biases: Vector<T, N>,
    weights: Matrix<T, N, I>,
}

impl<T: Scalar, const I: usize, const N: usize> Dense<T, I, N> {
    pub fn new(weights: Matrix<T, N, I>, biases: Vector<T, N>) -> Self {
        Dense { biases, weights }
    }
}

impl<T: Scalar, const I: usize, const N: usize> Layer<T, I, N>
    for Dense<T, I, N>
{
    fn forward(&self, inputs: &Vector<T, I>) -> Vector<T, N> {
        &self.weights * inputs + self.biases
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn forward() {
        let weights = Matrix([[1, 2, 3], [4, 5, 6]]);
        let biases = Vector([1, -1]);
        let layer = Dense::new(weights, biases);

        let inputs = Vector([4, 3, 2]);
        let outputs = layer.forward(&inputs);

        assert_eq!(Vector([4 + 6 + 6 + 1, 16 + 15 + 12 - 1]), outputs);
    }
}
