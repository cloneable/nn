use crate::math::{Scalar, Tensor, Vector};

pub trait ActivationFn<const N: usize, S: Scalar = f32> {
    fn apply(&self, inputs: &Vector<N, S>) -> Vector<N, S>;
}

pub struct ReLU;

impl<const N: usize, S: Scalar> ActivationFn<N, S> for ReLU {
    fn apply(&self, inputs: &Vector<N, S>) -> Vector<N, S> {
        let mut outputs = Vector::zero();
        for n in 0..N {
            outputs.0[n] = S::max(S::ZERO, inputs.0[n]);
        }
        outputs
    }
}

pub struct Softmax;

impl<const N: usize, S: Scalar> ActivationFn<N, S> for Softmax {
    fn apply(&self, inputs: &Vector<N, S>) -> Vector<N, S> {
        inputs.exp().norm()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn relu() {
        let inputs = [0.0, 0.2, -0.2].into();
        assert_eq!(Vector([0.0, 0.2, 0.0]), ReLU.apply(&inputs));
    }

    #[test]
    fn softmax() {
        let inputs = [2., 4., 1.].into();
        assert_eq!(
            Vector([0.1141952, 0.8437947, 0.042010065]),
            Softmax.apply(&inputs)
        );
    }
}
