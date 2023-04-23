use nn::{
    activation::{ActivationFn, ReLU, Softmax},
    layer::{Dense, Layer},
};

fn main() {
    let layer1 = Dense::new(
        &[[0.1, -0.2, 1.3], [1.1, -2.0, 0.8]].into(),
        [1.0, -1.0].into(),
    );
    let layer2 =
        Dense::new(&[[0.5, 0.1], [2.2, 0.2]].into(), [0.0, 0.0].into());

    let inputs = [[1.1, 2.2, 3.3]].into();
    let outputs = layer1.forward(&inputs);
    let outputs = ReLU.apply(&outputs);
    let outputs = layer2.forward(&outputs);
    let outputs = Softmax.apply(&outputs);

    println!("{outputs:?}");
}
