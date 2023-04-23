use crate::math::{tensor::Tensor, Scalar, Vector};
use std::ops;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Matrix<const R: usize, const C: usize, S: Scalar = f32>(
    pub [[S; C]; R],
);

impl<const R: usize, const C: usize, S: Scalar> Tensor<2, S>
    for Matrix<R, C, S>
{
    fn zero() -> Self {
        Matrix([[S::ZERO; C]; R])
    }

    fn shape(&self) -> [usize; 2] {
        [R, C]
    }

    fn get(&self, indices: [usize; 2]) -> S {
        self.0[indices[0]][indices[1]]
    }

    fn set(&mut self, indices: [usize; 2], value: S) {
        self.0[indices[0]][indices[1]] = value;
    }
}

impl<const R: usize, const C: usize, S: Scalar> From<[[S; C]; R]>
    for Matrix<R, C, S>
{
    fn from(v: [[S; C]; R]) -> Matrix<R, C, S> {
        Matrix(v)
    }
}

impl<const R: usize, const C: usize, S: Scalar> Matrix<R, C, S> {
    pub fn transpose(&self) -> Matrix<C, R, S> {
        let mut m = Matrix::zero();
        for r in 0..R {
            for c in 0..C {
                m.0[c][r] = self.0[r][c];
            }
        }
        m
    }
}

impl<const R1C0: usize, const R0: usize, const C1: usize, S: Scalar>
    ops::Mul<&Matrix<R1C0, C1, S>> for &Matrix<R0, R1C0, S>
{
    type Output = Matrix<R0, C1, S>;

    fn mul(self, rhs: &Matrix<R1C0, C1, S>) -> Self::Output {
        let mut m = Matrix::zero();
        for r in 0..R0 {
            for c in 0..C1 {
                let mut sum = S::zero();
                for rc in 0..R1C0 {
                    sum += self.0[r][rc] * rhs.0[rc][c];
                }
                m.0[r][c] = sum;
            }
        }
        m
    }
}

impl<const R: usize, const C: usize, S: Scalar> ops::Mul<&Vector<C, S>>
    for &Matrix<R, C, S>
{
    type Output = Vector<R, S>;

    fn mul(self, rhs: &Vector<C, S>) -> Self::Output {
        let mut v = Vector::zero();
        for r in 0..R {
            for c in 0..C {
                v.0[r] += self.0[r][c] * rhs.0[c];
            }
        }
        v
    }
}

impl<const R: usize, const C: usize, S: Scalar> ops::Mul<S>
    for &Matrix<R, C, S>
{
    type Output = Matrix<R, C, S>;

    fn mul(self, rhs: S) -> Self::Output {
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
    fn matrix_mul_matrix() {
        let m1 = Matrix([[3., 3., 3., 3.], [4., 4., 4., 4.]]);
        let m2 = Matrix([[2., 2.], [3., 3.], [4., 4.], [5., 5.]]);
        let m3 = &m1 * &m2;
        let m4 = &m2 * &m1;

        assert_eq!(
            Matrix([[3. * 2. + 3. * 3. + 3. * 4. + 3. * 5., 42.], [56., 56.]]),
            m3
        );
        assert_eq!(
            Matrix([
                [14., 14., 14., 14.],
                [21., 21., 21., 21.],
                [28., 28., 28., 28.],
                [35., 35., 35., 35.]
            ]),
            m4
        );
    }

    #[test]
    fn matrix_mul_vector() {
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
