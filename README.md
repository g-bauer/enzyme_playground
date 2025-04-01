# Testing Enzyme for automatic differentiation

A small toy project to try out different features of [Enzyme](https://enzyme.mit.edu/index.fcgi/rust/motivation.html).

```bash
rustc --version --verbose
rustc 1.88.0-nightly (b9ea82b84 2025-03-30)
binary: rustc
commit-hash: b9ea82b84ad02b4a9fe9e513626e65fb3f540838
commit-date: 2025-03-30
host: aarch64-apple-darwin
release: 1.88.0-nightly
LLVM version: 20.1.1
```

## Application
Calculate hard-sphere Helmholtz energy contribution with PC-SAFT temperature dependent diameter. This is a moderately complex function that depends on 

- temperature $T$, 
- volume $V$, and 
- moles $\vec{N}$ for $C-1$ components ($\vec{N} = [n_0, n_1, \dots, n_C]$). 

It also requires a set of parameters $\{m_i, \sigma_i, \epsilon_i\}$ for each component $i$.

The function returns a scalar and the goal is to calculate the partial derivatives w.r.t. to $T, V, \vec{N}$ (not the parameters for now).

As reference, we use a generic implementation using `num-dual` (dual numbers library):

```rust
pub fn a<D: DualNum<f64> + Copy>(
    parameters: &Parameters,
    temperature: D,
    volume: D,
    moles: &[D],
) -> D { ... }
```

Reference values:
```rust
// parameters
m = 2.001829
sigma = 3.618353
epsilon_k = 208.1101

// thermodynamic conditions
t = 250.0
v = 1000.0
n = [1.0]

// results
a     = 0.4106104925988083
da_dt = -0.00013057308352981093
da_dv = -0.00043679624456275116
da_dn = 0.8474067371615595
```


## Case 1: Forward-mode AD using `Vec` and `ndarray::ArrayBase`

- File: `src/bin/case1.rs`
- Run with: `RUSTFLAGS="-Zautodiff=Enable" cargo +enzyme run --bin case1 --release`

We calculate the partial derivative of a function $a(T, V, \vec{N})$ using *forward* AD.

1. calculate derivative using num-dual `Dual64`.
2. calculate derivative via enzyme.
3. calculate derivative via enzyme, but put all arguments into a single slice.

### Encountered Problems
The curious thing is: if variant 3 is not used, variant 1 and 2 yield identical results.
If variant three is used (after 1 and 2), the result of 2 is wrong.
Wheter variant 3 is calculated can be triggered with the `if false` statement (`true` fails, `false` works).

Output variant 1 & 2 (`if false ...`):
```bash
[src/bin/case1.rs:117:5] da_dt_nd = Dual {
    re: 0.4106104925988083,
    eps: -0.00013057308352981093,
    f: PhantomData<f64>,
}
[src/bin/case1.rs:118:5] da_dt_enz = (
    0.4106104925988083,
    -0.00013057308352981096,
)
// asserts all good
```

Output 1, 2 & 3 (`if true ...`):
```bash
[src/bin/case1.rs:117:5] da_dt_nd = Dual {
    re: 0.4106104925988083,
    eps: -0.00013057308352981093,
    f: PhantomData<f64>,
}
[src/bin/case1.rs:118:5] da_dt_enz = (
    0.4106104925988083,
    1.694682901239589, # <----- was correct before!
)
[src/bin/case1.rs:133:9] &da_dt_enz_args = (
    0.4106104925988083,
    -0.00013057308352981093,
)

thread 'main' panicked at src/bin/case1.rs:138:5:
assert_relative_eq!(da_dt_nd.eps, da_dt_enz.1, epsilon = 1e-14)

    left  = -0.00013057308352981093
    right = 1.694682901239589 # <----- panics now, worked before
```

There are several other issues happening:
- `src/bin/case1.rs:61-64`: if `m`, `sigma` and `epsilon` are changed from `ndarray::ArrayBase` to `Vec`, the code no longer compiles
- `src/bin/case1.rs:72-73` Using references in iterations (sometimes?) seems to be an issue. In line 23 it works
```rust
let partial_density: Vec<D> = moles.iter().map(|&n| n / volume).collect(); // line 23 <- works
let partial_density: Vec<f64> = moles.iter().cloned().map(|n| n / volume).collect(); // line 72 <- works
let partial_density: Vec<f64> = moles.iter().map(|&n| n / volume).collect(); // line 73 <- crashes
```


## Case 2: Forward-mode AD using slices

- File: `src/bin/case2.rs`
- Run with: `RUSTFLAGS="-Zautodiff=Enable" cargo +enzyme run --bin case2 --release`

Tried to do additional experiments with `a_f64_args` from `src/bin/case1.rs`. Copied the same function into file `src/bin/case2.rs` (without the other functions which are not needed). 

### Encountered Problems

Calling the function identically to `case1` does no longer compile. Changing the `Vec` or `ArrayBase` to slices makes the code compile again.

```rust
// src/bin/case.2.rs

// Now this crashes as well ...
// let m = arr1(&[2.001829]);
// let sigma = arr1(&[3.618353]);
// let epsilon_k = arr1(&[208.1101]);

// ... as does this ...
// let m = vec![2.001829];
// let sigma = vec![3.618353];
// let epsilon_k = vec![208.1101];

// but this works
let m = &[2.001829];
let sigma = &[3.618353];
let epsilon_k = &[208.1101];
```

## Case 3 (no problems): Forward and reverse AD with slices

- File: `src/bin/case3.rs`
- Run with: `RUSTFLAGS="-Zautodiff=Enable" cargo +enzyme run --bin case3 --release`

Same as `case2` but with slices. Works for forward and reverse.

```rust
#[autodiff(da_f, Forward, Dual, Dual)]
#[autodiff(da_r, Reverse, Duplicated, Active)]
pub fn a(args: &[f64]) -> f64 { ... }
```

## Case 4: Reverse AD with Active and Duplicated arguments 

- File: `src/bin/case3.rs`
- Run with: `RUSTFLAGS="-Zautodiff=Enable" cargo +enzyme run --bin case3 --release`

The Goal is to calculate the derivative w.r.t. all inputes besides `parameters`.

### Encountered Problems

Couldn't manage to make `Active` and `Duplicated` (of slice) work. This crashes:
```rust
#[autodiff(da_all_active, Reverse, Const, Active, Active, Duplicated, Active)]
pub fn helmholtz_energy(
    parameters: &Parameters,
    temperature: f64,
    volume: f64,
    moles: &[f64],
) -> f64 { ... }
```

Wrapping the above function inside a function that takes `&f64` instead of `f64` and using `Duplicated` instead of `Active` works:
```rust
#[autodiff(da_duplicated, Reverse, Const, Duplicated, Duplicated, Duplicated, Active)]
pub fn a_duplicated(
    parameters: &Parameters,
    temperature: &f64,
    volume: &f64,
    moles: &[f64],
) -> f64 {
    a_active(parameters, *temperature, *volume, moles)
}
```

