#![feature(autodiff)]
use approx::assert_relative_eq;
use std::autodiff::autodiff;
use std::f64::consts::FRAC_PI_6;

#[autodiff(da_f, Forward, Dual, Dual)]
#[autodiff(da_r, Reverse, Duplicated, Active)]
pub fn a(args: &[f64]) -> f64 {
    let temperature = args[0];
    let volume = args[1];
    let moles = &args[2..];

    let m = &[2.001829];
    let sigma = &[3.618353];
    let epsilon_k = &[208.1101];

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

fn main() {
    // reference values
    let a = 0.4106104925988083;
    let da_dt = -0.00013057308352981093;
    let da_dv = -0.00043679624456275116;
    let da_dn = 0.8474067371615595;

    let t = 250.0;
    let v = 1000.0;

    // forward
    let seed = &[1.0, 0.0, 0.0]; // first entry is temperature
    let da_dt_enz_args = da_f(&[t, v, 1.0], seed);
    dbg!(&da_dt_enz_args);

    // reverse
    let mut derivative = vec![0.0; 3];
    let seed_result = 1.0;
    let da_dt_enz_args = da_r(&[t, v, 1.0], &mut derivative, seed_result);
    dbg!(da_dt_enz_args);
    dbg!(&derivative);

    assert_relative_eq!(da_dt_enz_args, a, epsilon = 1e-14);
    assert_relative_eq!(derivative[0], da_dt, epsilon = 1e-14);
    assert_relative_eq!(derivative[1], da_dv, epsilon = 1e-14);
    assert_relative_eq!(derivative[2], da_dn, epsilon = 1e-14);
}
