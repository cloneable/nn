use crate::math::{Scalar, Vector};

pub trait ActivationFn<T: Scalar, const N: usize> {
    fn apply(&self, inputs: &Vector<T, N>) -> Vector<T, N>;
}
