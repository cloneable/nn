use crate::{
    layer::Layer,
    math::{Matrix, Scalar, Vector},
};

pub struct Dense<const I: usize, const N: usize, S: Scalar = f32> {
    biases: Vector<N, S>,
    weights: Matrix<I, N, S>,
}

impl<const I: usize, const N: usize, S: Scalar> Dense<I, N, S> {
    pub const fn new(weights: &Matrix<N, I, S>, biases: Vector<N, S>) -> Self {
        Dense { biases, weights: weights.transpose() }
    }
}

impl<
        const SAMPLES: usize,
        const INPUTS: usize,
        const OUTPUTS: usize,
        S: Scalar,
    > Layer<SAMPLES, INPUTS, OUTPUTS, S> for Dense<INPUTS, OUTPUTS, S>
{
    fn forward(
        &self,
        inputs: &Matrix<SAMPLES, INPUTS, S>,
    ) -> Matrix<SAMPLES, OUTPUTS, S> {
        &(inputs * &self.weights) + &self.biases
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn forward() {
        let weights = Matrix::new([[1., 2., 3.], [4., 5., 6.]]);
        let biases = Vector([1., -1.]);
        let layer = Dense::new(&weights, biases);

        let inputs = Matrix::new([[4., 3., 2.]]);
        let outputs = layer.forward(&inputs);

        assert_eq!(
            Matrix::new([[4. + 6. + 6. + 1., 16. + 15. + 12. - 1.]]),
            outputs
        );
    }
}
