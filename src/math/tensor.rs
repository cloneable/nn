pub trait Tensor<const RANK: usize, T> {
    const SHAPE: [usize; RANK];
}

pub struct Tensor3<T, const A: usize, const B: usize, const C: usize> {
    data: [[[T; C]; B]; A],
}

impl<T, const A: usize, const B: usize, const C: usize> Tensor<3, T>
    for Tensor3<T, A, B, C>
{
    const SHAPE: [usize; 3] = [A, B, C];
}

pub struct Tensor4<
    T,
    const A: usize,
    const B: usize,
    const C: usize,
    const D: usize,
> {
    data: [[[[T; D]; C]; B]; A],
}

impl<T, const A: usize, const B: usize, const C: usize, const D: usize>
    Tensor<4, T> for Tensor4<T, A, B, C, D>
{
    const SHAPE: [usize; 4] = [A, B, C, D];
}
