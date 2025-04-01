#![feature(autodiff)]
use approx::assert_relative_eq;
use std::autodiff::autodiff;
use std::f64::consts::FRAC_PI_6;

#[autodiff(da_active, Reverse, Const, Active, Active, Duplicated, Active)] // <- crashes
pub fn a_active(
    parameters: &Parameters,
    temperature: f64,
    volume: f64,
    moles: &[f64],
) -> f64 {
    let p = &parameters;
    let n = moles.len();
    let t_inv = temperature.recip();
    let diameter: Vec<f64> = (0..n)
        .map(|i| -((t_inv * -3.0 * p.epsilon_k[i]).exp() * 0.12 - 1.0) * p.sigma[i])
        .collect();

    let partial_density: Vec<f64> = moles.iter().cloned().map(|n| n / volume).collect(); // <- works
    // let partial_density: Vec<f64> = moles.iter().map(|&n| n / volume).collect(); // <- crashes
    let density: f64 = partial_density.iter().cloned().sum();
    let total_moles: f64 = moles.iter().cloned().sum();
    let x: Vec<f64> = moles.iter().cloned().map(|n| n / total_moles).collect();

    let mut zeta = [0.0; 4];
    for i in 0..diameter.len() {
        for (z, &k) in zeta.iter_mut().zip([0, 1, 2, 3].iter()) {
            *z += x[i] * diameter[i].powi(k) * (p.m[i] * FRAC_PI_6);
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

#[autodiff(da_duplicated, Reverse, Const, Duplicated, Duplicated, Duplicated, Active)]
pub fn a_duplicated(
    parameters: &Parameters,
    temperature: &f64,
    volume: &f64,
    moles: &[f64],
) -> f64 {
    a_active(parameters, *temperature, *volume, moles)
}

pub struct Parameters {
    m: Vec<f64>,
    sigma: Vec<f64>,
    epsilon_k: Vec<f64>,
}

fn main() {
    // reference values
    let a = 0.4106104925988083;
    let da_dt = -0.00013057308352981093;
    let da_dv = -0.00043679624456275116;
    let da_dn = 0.8474067371615595;

    let t = 250.0;
    let v = 1000.0;
    let n = &[1.0];
    let parameters = Parameters {
        m: vec![2.001829],
        sigma: vec![3.618353],
        epsilon_k: vec![208.1101],
    };

    // This works:
    // use Duplicated for float arguments 
    let mut da_dn_ = vec![0.0];
    let mut da_dt_ = 0.0;
    let mut da_dv_ = 0.0;
    let all_deritvatives = da_duplicated(&parameters, &t, &mut da_dt_, &v, &mut da_dv_, n, &mut da_dn_, 1.0);
    dbg!(all_deritvatives);
    dbg!(da_dt_);
    dbg!(da_dv_);
    dbg!(da_dn_);

    // This crashes if set to true:
    // use Active for float arguments and Duplicated for slice
    if false {
        let mut da_dn_ = vec![0.0];
        let all_deritvatives = da_active(&parameters, t, v, n, &mut da_dn_, 1.0);
        dbg!(all_deritvatives);
        dbg!(da_dn_);
    }
}
