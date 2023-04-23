use crate::math::{Matrix, Scalar};

pub trait ActivationFn<
    const SAMPLES: usize,
    const OUTPUTS: usize,
    S: Scalar = f32,
>
{
    fn apply(
        &self,
        outputs: &Matrix<SAMPLES, OUTPUTS, S>,
    ) -> Matrix<SAMPLES, OUTPUTS, S>;
}

pub struct ReLU;

impl<const SAMPLES: usize, const OUTPUTS: usize, S: Scalar>
    ActivationFn<SAMPLES, OUTPUTS, S> for ReLU
{
    fn apply(
        &self,
        outputs: &Matrix<SAMPLES, OUTPUTS, S>,
    ) -> Matrix<SAMPLES, OUTPUTS, S> {
        let mut m = Matrix::zero();
        for s in 0..SAMPLES {
            for o in 0..OUTPUTS {
                m.0[s].0[o] = S::max(S::ZERO, outputs.0[s].0[o]);
            }
        }
        m
    }
}

pub struct Softmax;

impl<const SAMPLES: usize, const OUTPUTS: usize, S: Scalar>
    ActivationFn<SAMPLES, OUTPUTS, S> for Softmax
{
    fn apply(
        &self,
        outputs: &Matrix<SAMPLES, OUTPUTS, S>,
    ) -> Matrix<SAMPLES, OUTPUTS, S> {
        let mut m = Matrix::zero();
        for s in 0..SAMPLES {
            m.0[s] = outputs.0[s].exp().norm();
        }
        m
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn relu() {
        let inputs = [[0.0, 0.2, -0.2]].into();
        assert_eq!(Matrix::new([[0.0, 0.2, 0.0]]), ReLU.apply(&inputs));
    }

    #[test]
    fn softmax() {
        let inputs = [[2., 4., 1.]].into();
        assert_eq!(
            Matrix::new([[0.1141952, 0.8437947, 0.042010065]]),
            Softmax.apply(&inputs)
        );
    }
}
