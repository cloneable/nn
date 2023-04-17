use std::ops;

pub trait Scalar:
    ops::Add<Output = Self>
    + ops::AddAssign
    + ops::Mul<Output = Self>
    + ops::MulAssign
    + ops::DivAssign
    + Copy
    + PartialEq
    + PartialOrd
{
    const ZERO: Self;

    fn max(self, other: Self) -> Self {
        if self >= other {
            self
        } else {
            other
        }
    }

    fn exp(self) -> Self;

    fn abs(self) -> Self;
}

impl Scalar for f32 {
    const ZERO: Self = 0.0;

    fn exp(self) -> Self {
        f32::exp(self)
    }

    fn abs(self) -> Self {
        f32::abs(self)
    }
}
