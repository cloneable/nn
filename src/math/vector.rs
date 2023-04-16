use crate::math::Scalar;

#[derive(Copy, Clone, Debug, PartialEq)]
pub struct Vector<T: Scalar, const N: usize>(pub [T; N]);

impl<const N: usize, T: Scalar> Vector<T, N> {
    pub const fn zero() -> Self {
        Vector([T::ZERO; N])
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
}
