use crate::math::Scalar;
use std::fmt::Debug;

pub trait Tensor<const RANK: usize, S: Scalar = f32>: Debug {
    fn zero() -> Self;
    fn shape(&self) -> [usize; RANK];
    fn get(&self, indices: [usize; RANK]) -> S;
    fn set(&mut self, indices: [usize; RANK], value: S);

    // TOOD: access lower ranked tensors
}

#[derive(Debug)]
pub struct Tensor3<
    const A: usize,
    const B: usize,
    const C: usize,
    S: Scalar = f32,
>(pub [[[S; C]; B]; A]);

impl<S: Scalar, const A: usize, const B: usize, const C: usize> Tensor<3, S>
    for Tensor3<A, B, C, S>
{
    fn zero() -> Self {
        Tensor3([[[S::ZERO; C]; B]; A])
    }

    fn shape(&self) -> [usize; 3] {
        [A, B, C]
    }

    fn get(&self, indices: [usize; 3]) -> S {
        self.0[indices[0]][indices[1]][indices[2]]
    }

    fn set(&mut self, indices: [usize; 3], value: S) {
        self.0[indices[0]][indices[1]][indices[2]] = value;
    }
}

impl<const A: usize, const B: usize, const C: usize, S: Scalar>
    From<[[[S; C]; B]; A]> for Tensor3<A, B, C, S>
{
    fn from(v: [[[S; C]; B]; A]) -> Tensor3<A, B, C, S> {
        Tensor3(v)
    }
}
