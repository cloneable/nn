mod dense;

pub use dense::Dense;

use crate::math::{Scalar, Vector};

pub trait Layer<const I: usize, const O: usize, S: Scalar = f32> {
    fn forward(&self, inputs: &Vector<I, S>) -> Vector<O, S>;
}
