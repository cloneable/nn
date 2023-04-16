use crate::math::Scalar;

#[derive(Copy, Clone, Debug)]
pub struct Vector<T: Scalar, const N: usize>(pub [T; N]);

impl<const N: usize, T: Scalar> Vector<T, N> {
    pub const fn zero() -> Self {
        Vector([T::ZERO; N])
    }
}
