use crate::math::{Scalar, Vector};
use std::ops;

#[derive(Clone, Debug)]
pub struct Matrix<T: Scalar, const R: usize, const C: usize>(pub [[T; C]; R]);

impl<const R: usize, const C: usize, T: Scalar> Matrix<T, R, C> {
    pub const fn zero() -> Self {
        Matrix([[T::ZERO; C]; R])
    }

    pub fn transpose(&self) -> Matrix<T, C, R> {
        let mut m = Matrix::zero();
        for r in 0..R {
            for c in 0..C {
                m.0[c][r] = self.0[r][c];
            }
        }
        m
    }
}

impl<const R: usize, const C: usize, T: Scalar> ops::Mul<&Vector<T, C>>
    for &Matrix<T, R, C>
{
    type Output = Vector<T, R>;

    fn mul(self, rhs: &Vector<T, C>) -> Self::Output {
        let mut v = Vector::zero();
        for r in 0..R {
            for c in 0..C {
                v.0[r] += self.0[r][c] * rhs.0[c];
            }
        }
        v
    }
}

impl<const R: usize, const C: usize, T: Scalar> ops::Mul<T>
    for &Matrix<T, R, C>
{
    type Output = Matrix<T, R, C>;

    fn mul(self, rhs: T) -> Self::Output {
        let mut m = self.clone();
        for r in 0..R {
            for c in 0..C {
                m.0[c][r] *= rhs;
            }
        }
        m
    }
}
