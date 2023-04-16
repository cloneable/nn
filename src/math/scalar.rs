use std::ops;

pub trait Scalar:
    ops::Add<Output = Self>
    + ops::AddAssign
    + ops::Mul<Output = Self>
    + ops::MulAssign
    + Copy
    + PartialEq
{
    const ZERO: Self;
}

impl Scalar for i32 {
    const ZERO: Self = 0;
}

impl Scalar for f32 {
    const ZERO: Self = 0.0;
}
