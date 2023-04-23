mod dense;

use crate::math::{Matrix, Scalar};
pub use dense::Dense;

pub trait Layer<
    const SAMPLES: usize,
    const INPUTS: usize,
    const OUTPUTS: usize,
    S: Scalar = f32,
>
{
    fn forward(
        &self,
        inputs: &Matrix<SAMPLES, INPUTS, S>,
    ) -> Matrix<SAMPLES, OUTPUTS, S>;
}
