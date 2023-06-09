use crate::math::Tensor;
use rand::{distributions::Standard, Rng};
use std::ops;

pub trait Scalar:
    Tensor<0, Self>
    + ops::Add<Output = Self>
    + ops::AddAssign
    + ops::Mul<Output = Self>
    + ops::MulAssign
    + ops::Div<Output = Self>
    + ops::DivAssign
    + Copy
    + PartialEq
    + PartialOrd
    + Default
{
    const ZERO: Self;

    #[must_use]
    fn max(self, other: Self) -> Self {
        if self >= other {
            self
        } else {
            other
        }
    }

    #[must_use]
    fn exp(self) -> Self;

    #[must_use]
    fn abs(self) -> Self;

    #[must_use]
    fn from_usize(v: usize) -> Self;

    #[must_use]
    fn random<RNG: Rng + ?Sized>(rng: &mut RNG) -> Self;
}

impl<S: Scalar> Tensor<0, S> for S {
    fn shape(&self) -> [usize; 0] {
        []
    }

    fn get(&self, _indices: [usize; 0]) -> S {
        *self
    }

    fn set(&mut self, _indices: [usize; 0], value: S) {
        *self = value;
    }
}

impl Scalar for f32 {
    const ZERO: Self = 0.0;

    fn exp(self) -> Self {
        f32::exp(self)
    }

    fn abs(self) -> Self {
        f32::abs(self)
    }

    #[allow(clippy::cast_precision_loss)]
    fn from_usize(v: usize) -> Self {
        v as f32
    }

    fn random<RNG: Rng + ?Sized>(rng: &mut RNG) -> Self {
        rng.sample(Standard)
    }
}
