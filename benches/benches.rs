#![feature(autodiff)]

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use num_dual::{Dual64, Dual2_64, DualNum};

pub fn f1<D: DualNum<f64>>(x: D) -> D {
    x.exp() / (x.sin().powi(3) + x.cos().powi(3)).sqrt()
}

#[autodiff(df1, Forward, Dual, Dual)]
pub fn _f1(x: f64) -> f64 {
    f1(x)
}

// This crashes:
// #[autodiff(d2f1, Forward, Dual)]
// pub fn _df1(x: f64) {
//     let _ = df1(x, 1.0);
// }


pub fn benchmark(c: &mut Criterion) {
    let x = 1.5;
    // make sure that it actually works
    dbg!(df1(x, 1.0));

    c.bench_with_input(BenchmarkId::new("Enzyme: 1st order", "forward"), &x, |b, &x| {
        b.iter(|| df1(x, 1.0));
    });

    // c.bench_with_input(BenchmarkId::new("Enzyme: 2st order", "forward"), &x, |b, &x| {
    //     b.iter(|| d2f1(x, 1.0));
    // });

    let xd = Dual64::new(x, 1.0);
    c.bench_with_input(
        BenchmarkId::new("num-dual: 1st order", "Dual64"),
        &xd,
        |b, &x| {
            b.iter(|| f1(x));
        },
    );

    let xd2 = Dual2_64::from_re(x).derivative();
    c.bench_with_input(
        BenchmarkId::new("num-dual: 2st order", "Dual2_64"),
        &xd2,
        |b, &x| {
            b.iter(|| f1(x));
        },
    );
}

criterion_group!(benches, benchmark); //enzyme_benchmark, num_dual_benchmark);
criterion_main!(benches);
