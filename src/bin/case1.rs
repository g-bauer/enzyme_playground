#![feature(autodiff)]
use std::autodiff::autodiff;
use std::f64::consts::FRAC_PI_6;

use approx::assert_relative_eq;
use ndarray::arr1;
use num_dual::{Dual64, DualNum};

pub fn a<D: DualNum<f64> + Copy>(
    parameters: &Parameters,
    temperature: D,
    volume: D,
    moles: &[D],
) -> D {
    let n = moles.len();
    let t_inv = temperature.recip();
    let diameter: Vec<D> = (0..n)
        .map(|i| {
            -((t_inv * -3.0 * parameters.epsilon_k[i]).exp() * 0.12 - 1.0) * parameters.sigma[i]
        })
        .collect();

    let partial_density: Vec<D> = moles.iter().map(|&n| n / volume).collect();
    let density: D = partial_density.iter().cloned().sum();
    let total_moles: D = moles.iter().cloned().sum();
    let x: Vec<D> = moles.iter().map(|&n| n / total_moles).collect();

    let mut zeta = [D::zero(); 4];
    for i in 0..diameter.len() {
        for (z, &k) in zeta.iter_mut().zip([0, 1, 2, 3].iter()) {
            *z += x[i] * diameter[i].powi(k) * (parameters.m[i] * FRAC_PI_6);
        }
    }
    let zeta_23 = zeta[2] / zeta[3];

    zeta.iter_mut().for_each(|z| *z *= density);
    let frac_1mz3 = -(zeta[3] - 1.0).recip();
    let a = volume / std::f64::consts::FRAC_PI_6
        * (zeta[1] * zeta[2] * frac_1mz3 * 3.0
            + zeta[2].powi(2) * frac_1mz3.powi(2) * zeta_23
            + (zeta[2] * zeta_23.powi(2) - zeta[0]) * (zeta[3] * (-1.0)).ln_1p());
    a
}

#[autodiff(da, Forward, Const, Dual, Dual, Const, Dual)]
pub fn a_f64(parameters: &Parameters, temperature: f64, volume: f64, moles: &[f64]) -> f64 {
    a(parameters, temperature, volume, moles)
}

#[autodiff(da_args, Forward, Dual, Dual)]
pub fn a_f64_args(args: &[f64]) -> f64 {
    let temperature = args[0];
    let volume = args[1];
    let moles = &args[2..];

    // This works
    let m = arr1(&[2.001829]);
    let sigma = arr1(&[3.618353]);
    let epsilon_k = arr1(&[208.1101]);

    // changing to vec crashes
    // let m = vec![2.001829];
    // let sigma = vec![3.618353];
    // let epsilon_k = vec![208.1101];

    let n = moles.len();
    let t_inv = temperature.recip();
    let diameter: Vec<f64> = (0..n)
        .map(|i| -((t_inv * -3.0 * epsilon_k[i]).exp() * 0.12 - 1.0) * sigma[i])
        .collect();

    let partial_density: Vec<f64> = moles.iter().cloned().map(|n| n / volume).collect(); // <- works
    // let partial_density: Vec<f64> = moles.iter().map(|&n| n / volume).collect(); // <- crashes
    let density: f64 = partial_density.iter().cloned().sum();
    let total_moles: f64 = moles.iter().cloned().sum();
    let x: Vec<f64> = moles.iter().cloned().map(|n| n / total_moles).collect();

    let mut zeta = [0.0; 4];
    for i in 0..diameter.len() {
        for (z, &k) in zeta.iter_mut().zip([0, 1, 2, 3].iter()) {
            *z += x[i] * diameter[i].powi(k) * (m[i] * FRAC_PI_6);
        }
    }
    let zeta_23 = zeta[2] / zeta[3];

    zeta.iter_mut().for_each(|z| *z *= density);
    let frac_1mz3 = -(zeta[3] - 1.0).recip();
    let a = volume / std::f64::consts::FRAC_PI_6
        * (zeta[1] * zeta[2] * frac_1mz3 * 3.0
            + zeta[2].powi(2) * frac_1mz3.powi(2) * zeta_23
            + (zeta[2] * zeta_23.powi(2) - zeta[0]) * (zeta[3] * (-1.0)).ln_1p());
    a
}

pub struct Parameters {
    m: Vec<f64>,
    sigma: Vec<f64>,
    epsilon_k: Vec<f64>,
}

fn main() {
    let t = 250.0;
    let v = 1000.0;
    let n = &[1.0];
    let parameters = Parameters {
        m: vec![2.001829],
        sigma: vec![3.618353],
        epsilon_k: vec![208.1101],
    };

    // Calculate da_dt using num-dual
    let da_dt_nd = a(&parameters, Dual64::new(t, 1.0), v.into(), &[n[0].into()]);

    // Calculate da_dt using enzyme
    let da_dt_enz = da(&parameters, t, 1.0, v, 0.0, n);
    
    dbg!(da_dt_nd);
    dbg!(da_dt_enz);

    // if set to false:
    // - the above derivative da_dt_enz is correct
    // - compiler message: freeing without malloc   %128 = phi ptr [ %105, %118 ], [ %105, %104 ], [ inttoptr (i64 8 to ptr), %23 ]

    // if set to true:
    // - the above derivative da_dt_enz is wrong
    // - compiler message: 
    // freeing without malloc   %27 = select i1 false, ptr inttoptr (i64 8 to ptr), ptr %26
    // freeing without malloc   %24 = select i1 false, ptr inttoptr (i64 8 to ptr), ptr %23
    // freeing without malloc   %21 = tail call noundef dereferenceable_or_null(8) ptr @malloc(i64 noundef range(i64 1, 0) 8) #91
    if true {
        let seed = &[1.0, 0.0, 0.0]; // first entry is temperature
        let da_dt_enz_args = da_args(&[t, v, 1.0], seed);
        dbg!(&da_dt_enz_args);
    }

    // compare dual number and enzyme derivative
    assert_relative_eq!(da_dt_nd.re, da_dt_enz.0, epsilon = 1e-14);
    assert_relative_eq!(da_dt_nd.eps, da_dt_enz.1, epsilon = 1e-14);
}
