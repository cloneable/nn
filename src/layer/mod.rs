mod dense;

pub use dense::Dense;

use crate::math::{Scalar, Vector};

pub trait Layer<T: Scalar, const I: usize, const O: usize> {
    fn forward(&self, inputs: &Vector<T, I>) -> Vector<T, O>;
}
