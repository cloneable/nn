use crate::{
    layer::Layer,
    math::{Matrix, Scalar, Vector},
};

pub struct Dense<const I: usize, const N: usize, S: Scalar = f32> {
    biases: Vector<N, S>,
    weights: Matrix<N, I, S>,
}

impl<const I: usize, const N: usize, S: Scalar> Dense<I, N, S> {
    pub const fn new(weights: Matrix<N, I, S>, biases: Vector<N, S>) -> Self {
        Dense { biases, weights }
    }
}

impl<const I: usize, const N: usize, S: Scalar> Layer<I, N, S>
    for Dense<I, N, S>
{
    fn forward(&self, inputs: &Vector<I, S>) -> Vector<N, S> {
        &self.weights * inputs + self.biases
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn forward() {
        let weights = Matrix([[1., 2., 3.], [4., 5., 6.]]);
        let biases = Vector([1., -1.]);
        let layer = Dense::new(weights, biases);

        let inputs = Vector([4., 3., 2.]);
        let outputs = layer.forward(&inputs);

        assert_eq!(Vector([4. + 6. + 6. + 1., 16. + 15. + 12. - 1.]), outputs);
    }
}
