use std::f64::consts::FRAC_PI_6;
use num_dual::{Dual64, DualNum};

pub struct Parameters {
    m: Vec<f64>,
    sigma: Vec<f64>,
    epsilon_k: Vec<f64>,
}

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
    let a64 = a(&parameters, t, v, n);
    let da_dt = a(&parameters, Dual64::new(t, 1.0), v.into(), &[n[0].into()]);
    let da_dv = a(&parameters, t.into(), Dual64::new(v, 1.0), &[n[0].into()]);
    let da_dn = a(&parameters, t.into(), v.into(), &[Dual64::new(n[0], 1.0)]);
    dbg!(a64, da_dt, da_dv, da_dn);
}
