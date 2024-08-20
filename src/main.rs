#![feature(autodiff)]
use enzyme_playground::{df1_lib, f1};
use num_dual::Dual64;

#[autodiff(df1r, Reverse, Active, Active)]
#[autodiff(df1f, Forward, Dual, Dual)]
pub fn _f1(x: f64) -> f64 {
    f1(x)
}

/// Second order forward derivative
// #[autodiff(df1, Forward, Dual, Dual)]
// pub fn _f2(x: f64) -> ? {
//     df1(x, 1.0)
// }

fn main() {
    // input for enzyme
    let (x, seed) = (1.5, 1.0);

    // num-dual for comparison
    let dual_y1 = f1(Dual64::new(x, seed));

    // call function defined in lib.rs
    let enzyme_y1_lib = df1_lib(x, seed);

    // call function defined above main.
    let enzyme_y1f = df1f(x, seed);

    // this crashes (follows: https://enzyme.mit.edu/index.fcgi/rust/user_design.html#design)
    // let enzyme_y1r = df1r(x, seed);

    dbg!(dual_y1);
    dbg!(enzyme_y1_lib);
    dbg!(enzyme_y1f);
}
