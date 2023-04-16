use std::ops;

pub trait Scalar:
    ops::Add<Output = Self>
    + ops::AddAssign
    + ops::Mul<Output = Self>
    + ops::MulAssign
    + Copy
{
    const ZERO: Self;
}

impl Scalar for f32 {
    const ZERO: Self = 0.0;
}
