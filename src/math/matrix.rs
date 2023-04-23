use crate::math::{tensor::Tensor, Scalar, Vector};
use std::ops;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Matrix<const R: usize, const C: usize, S: Scalar = f32>(
    pub [Vector<C, S>; R],
);

impl<const R: usize, const C: usize, S: Scalar> Tensor<2, S>
    for Matrix<R, C, S>
{
    fn shape(&self) -> [usize; 2] {
        [R, C]
    }

    fn get(&self, indices: [usize; 2]) -> S {
        self.0[indices[0]].0[indices[1]]
    }

    fn set(&mut self, indices: [usize; 2], value: S) {
        self.0[indices[0]].0[indices[1]] = value;
    }
}

impl<const R: usize, const C: usize, S: Scalar> From<[[S; C]; R]>
    for Matrix<R, C, S>
{
    fn from(v: [[S; C]; R]) -> Self {
        Matrix::new(v)
    }
}

impl<const R: usize, const C: usize, S: Scalar> Matrix<R, C, S> {
    pub const fn new(v: [[S; C]; R]) -> Self {
        let mut m = Matrix::zero();
        let mut r = 0;
        while r < R {
            m.0[r] = Vector(v[r]);
            r += 1;
        }
        m
    }

    pub const fn zero() -> Self {
        Matrix([Vector::zero(); R])
    }

    pub const fn transpose(&self) -> Matrix<C, R, S> {
        let mut m = Matrix::zero();
        let mut r = 0;
        while r < R {
            let mut c = 0;
            while c < C {
                m.0[c].0[r] = self.0[r].0[c];
                c += 1;
            }
            r += 1;
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
                let mut sum = S::ZERO;
                for rc in 0..R1C0 {
                    sum += self.0[r].0[rc] * rhs.0[rc].0[c];
                }
                m.0[r].0[c] = sum;
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
                v.0[r] += self.0[r].0[c] * rhs.0[c];
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
                m.0[r].0[c] *= rhs;
            }
        }
        m
    }
}

impl<const R: usize, const C: usize, S: Scalar> ops::Add<&Vector<C, S>>
    for &Matrix<R, C, S>
{
    type Output = Matrix<R, C, S>;

    fn add(self, rhs: &Vector<C, S>) -> Self::Output {
        let mut m = Matrix::zero();
        for r in 0..R {
            m.0[r] = self.0[r] + *rhs;
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
        assert_eq!(Matrix::new([[0., 0., 0.], [0., 0., 0.]]), m);
    }

    #[test]
    fn matrix_transpose() {
        let m = Matrix::new([[1., 2., 3.], [4., 5., 6.]]);
        assert_eq!(
            Matrix::new([[1., 4.], [2., 5.].into(), [3., 6.].into()]),
            m.transpose()
        );
    }

    #[test]
    fn matrix_mul_matrix() {
        let m1 = Matrix::new([[3., 3., 3., 3.], [4., 4., 4., 4.]]);
        let m2 = Matrix::new([[2., 2.], [3., 3.], [4., 4.], [5., 5.]]);
        let m3 = &m1 * &m2;
        let m4 = &m2 * &m1;

        assert_eq!(
            Matrix::new([
                [3. * 2. + 3. * 3. + 3. * 4. + 3. * 5., 42.],
                [56., 56.]
            ]),
            m3
        );
        assert_eq!(
            Matrix::new([
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
        let m = Matrix::new([[2., 2., 2.], [3., 3., 3.]]);
        let v = Vector([1., 2., 3.]);
        assert_eq!(Vector([2. + 4. + 6., 3. + 6. + 9.]), &m * &v);
    }

    #[test]
    fn matrix_mul_scalar() {
        let m = Matrix::new([[1., 2., 3.], [4., 5., 6.]]);
        let s = 3.;
        assert_eq!(Matrix::new([[3., 6., 9.], [12., 15., 18.]]), &m * s);
    }

    #[test]
    fn matrix_add_vector() {
        let m = Matrix::new([[1., 2., 3.], [4., 5., 6.]]);
        let v = Vector::new([2., 2., 2.]);
        assert_eq!(Matrix::new([[3., 4., 5.], [6., 7., 8.]]), &m + &v);
    }
}
