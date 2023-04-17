use crate::math::{Scalar, Vector};
use std::ops;

#[derive(Clone, Debug, PartialEq)]
pub struct Matrix<T: Scalar, const R: usize, const C: usize>(pub [[T; C]; R]);

impl<T: Scalar, const R: usize, const C: usize> Into<Matrix<T, R, C>>
    for [[T; C]; R]
{
    fn into(self) -> Matrix<T, R, C> {
        Matrix(self)
    }
}

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
                m.0[r][c] *= rhs;
            }
        }
        m
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn matrix_zero() {
        let m = Matrix::zero();
        assert_eq!(Matrix([[0., 0., 0.], [0., 0., 0.]]), m);
    }

    #[test]
    fn matrix_transpose() {
        let m = Matrix([[1., 2., 3.], [4., 5., 6.]]);
        assert_eq!(Matrix([[1., 4.], [2., 5.], [3., 6.]]), m.transpose());
    }

    #[test]
    fn matrix_dot_vector() {
        let m = Matrix([[2., 2., 2.], [3., 3., 3.]]);
        let v = Vector([1., 2., 3.]);
        assert_eq!(Vector([2. + 4. + 6., 3. + 6. + 9.]), &m * &v);
    }

    #[test]
    fn matrix_mul_scalar() {
        let m = Matrix([[1., 2., 3.], [4., 5., 6.]]);
        let s = 3.;
        assert_eq!(Matrix([[3., 6., 9.], [12., 15., 18.]]), &m * s);
    }
}
