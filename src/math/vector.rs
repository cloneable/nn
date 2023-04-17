use crate::math::{Scalar, Tensor};
use std::ops;

#[derive(Copy, Clone, Debug, PartialEq)]
pub struct Vector<T: Scalar, const N: usize>(pub [T; N]);

impl<T: Scalar, const N: usize> Tensor<1, T> for Vector<T, N> {
    const SHAPE: [usize; 1] = [N];
}

impl<T: Scalar, const N: usize> Into<Vector<T, N>> for [T; N] {
    fn into(self) -> Vector<T, N> {
        Vector(self)
    }
}

impl<T: Scalar, const N: usize> Vector<T, N> {
    pub const fn zero() -> Self {
        Vector([T::ZERO; N])
    }

    pub fn exp(mut self) -> Self {
        for n in &mut self.0 {
            *n = n.exp();
        }
        self
    }

    pub fn abs(mut self) -> Self {
        for n in &mut self.0 {
            *n = n.abs();
        }
        self
    }

    pub fn sum(&self) -> T {
        let mut sum = T::ZERO;
        for n in self.0 {
            sum += n;
        }
        sum
    }

    pub fn norm(mut self) -> Self {
        // TODO: opt: normalize without abs and 0 check.
        let sum = self.abs().sum();
        if sum == T::ZERO {
            return self;
        }
        for n in 0..N {
            self.0[n] /= sum;
        }
        self
    }
}

impl<T: Scalar, const N: usize> ops::Add for Vector<T, N> {
    type Output = Vector<T, N>;

    fn add(self, rhs: Vector<T, N>) -> Self::Output {
        let mut v = self.clone();
        for n in 0..N {
            v.0[n] += rhs.0[n];
        }
        v
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn vector_zero() {
        let v = Vector::zero();
        assert_eq!(Vector([0., 0., 0., 0.]), v);
    }

    #[test]
    fn vector_plus_vector() {
        let a = Vector([1., 2., 3., 4.]);
        let b = Vector([4., 3., 2., 1.]);
        assert_eq!(Vector([5., 5., 5., 5.]), a + b);
    }

    #[test]
    fn vector_exp() {
        let a = Vector([1., 2., 3.]);
        assert_eq!(Vector([2.7182817, 7.389056, 20.085537]), a.exp());
        assert_eq!(Vector([1., 2., 3.]), a);
    }

    #[test]
    fn vector_abs() {
        let a = Vector([-1., 2., -3.]);
        assert_eq!(Vector([1., 2., 3.]), a.abs());
        assert_eq!(Vector([-1., 2., -3.]), a);
    }

    #[test]
    fn vector_sum() {
        let a = Vector([1., 2., 3.]);
        assert_eq!(6., a.sum());
        assert_eq!(Vector([1., 2., 3.]), a);
    }

    #[test]
    fn vector_norm() {
        let a = Vector([1., 2., 3.]);
        assert_eq!(Vector([0.16666667, 0.33333334, 0.5]), a.norm());
        assert_eq!(Vector([1., 2., 3.]), a);

        let a = Vector([-1., -2., 3.]);
        assert_eq!(Vector([-0.16666667, -0.33333334, 0.5]), a.norm());
        assert_eq!(Vector([-1., -2., 3.]), a);
    }
}
