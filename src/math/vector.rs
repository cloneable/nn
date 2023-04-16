use crate::math::Scalar;
use std::ops;

#[derive(Copy, Clone, Debug, PartialEq)]
pub struct Vector<T: Scalar, const N: usize>(pub [T; N]);

impl<T: Scalar, const N: usize> Vector<T, N> {
    pub const fn zero() -> Self {
        Vector([T::ZERO; N])
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
        assert_eq!(Vector([0, 0, 0, 0]), v);
    }

    #[test]
    fn vector_plus_vector() {
        let a = Vector([1, 2, 3, 4]);
        let b = Vector([4, 3, 2, 1]);
        assert_eq!(Vector([5, 5, 5, 5]), a + b);
    }
}
