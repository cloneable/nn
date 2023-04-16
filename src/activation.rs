use crate::math::{Scalar, Vector};

pub trait ActivationFn<T: Scalar, const N: usize> {
    fn apply(&self, inputs: &Vector<T, N>) -> Vector<T, N>;
}

pub struct ReLU();

impl<T: Scalar, const N: usize> ActivationFn<T, N> for ReLU {
    fn apply(&self, inputs: &Vector<T, N>) -> Vector<T, N> {
        let mut outputs = inputs.clone();
        for n in 0..N {
            outputs.0[n] = T::max(T::ZERO, outputs.0[n]);
        }
        outputs
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn relu() {
        let inputs = [0.0, 0.2, -0.2].into();
        assert_eq!(Vector([0.0, 0.2, 0.0]), ReLU().apply(&inputs));
    }
}
