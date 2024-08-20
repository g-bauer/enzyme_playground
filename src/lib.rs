#![feature(autodiff)]

use num_dual::DualNum;

pub fn f1<D: DualNum<f64>>(x: D) -> D {
    x.exp() / (x.sin().powi(3) + x.cos().powi(3)).sqrt()
}

#[autodiff(df1_lib, Forward, Dual, Dual)]
pub fn _f1(x: f64) -> f64 {
    f1(x)
}