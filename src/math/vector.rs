use crate::math::{Scalar, Tensor};
use std::ops;

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct Vector<const N: usize, S: Scalar = f32>(pub [S; N]);

impl<const N: usize, S: Scalar> Tensor<1, S> for Vector<N, S> {
    fn shape(&self) -> [usize; 1] {
        [N]
    }

    fn get(&self, indices: [usize; 1]) -> S {
        debug_assert!(indices[0] < N);
        self.0[indices[0]]
    }

    fn set(&mut self, indices: [usize; 1], value: S) {
        debug_assert!(indices[0] < N);
        self.0[indices[0]] = value;
    }
}

impl<const N: usize, S: Scalar> From<[S; N]> for Vector<N, S> {
    fn from(v: [S; N]) -> Self {
        Vector::new(v)
    }
}

impl<const N: usize, S: Scalar> Vector<N, S> {
    pub const fn new(v: [S; N]) -> Self {
        Vector(v)
    }

    pub const fn zero() -> Self {
        Vector([S::ZERO; N])
    }

    #[must_use]
    pub fn exp(mut self) -> Self {
        for n in &mut self.0 {
            *n = n.exp();
        }
        self
    }

    #[must_use]
    pub fn abs(mut self) -> Self {
        for n in &mut self.0 {
            *n = n.abs();
        }
        self
    }

    #[must_use]
    pub fn sum(&self) -> S {
        let mut sum = S::ZERO;
        for n in self.0 {
            sum += n;
        }
        sum
    }

    #[must_use]
    pub fn norm(mut self) -> Self {
        // TODO: opt: normalize without abs and 0 check.
        let sum = self.abs().sum();
        if sum == S::ZERO {
            return self;
        }
        for n in 0..N {
            self.0[n] /= sum;
        }
        self
    }

    #[must_use]
    pub fn mean(&self) -> S {
        if N == 0 {
            S::ZERO
        } else {
            self.sum() / S::from_usize(N)
        }
    }

    #[must_use]
    pub fn clip(mut self, min: S, max: S) -> Self {
        debug_assert!(min <= max);
        for n in &mut self.0 {
            *n = if *n < min {
                min
            } else if *n > max {
                max
            } else {
                *n
            };
        }
        self
    }
}

impl<const N: usize, S: Scalar> ops::Add for Vector<N, S> {
    type Output = Vector<N, S>;

    fn add(self, rhs: Vector<N, S>) -> Self::Output {
        let mut v = self;
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
