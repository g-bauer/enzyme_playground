# Testing Enzyme for automatic differentiation

A small toy project to try out different features of [Enzyme](https://enzyme.mit.edu/index.fcgi/rust/motivation.html).

## Problems

**Functions (with `autodiff` macro) imported from modules are not working properly.**

See `main.rs`

```bash
# num-dual for comparison
[src/main.rs:33:5] dual_y1 = Dual {
    re: 4.497780053946162,
    eps: 4.05342789389862,
    f: PhantomData<f64>,
}
# identical function as below, but defined in lib.rs
[src/main.rs:34:5] enzyme_y1_lib = (
    4.497780053946161,
    0.0, # <--- ?
)
# identical function as above, but defined in main.rs
[src/main.rs:35:5] enzyme_y1f = (
    4.497780053946161,
    4.05342789389862,
)
```

- Reverse-mode AD crashes (following example in documentation)
- Higher-order derivative using `Forward` twice crashes.

## Performance

A quick check against `num-dual` for the forward-mode first derivative. These cases should be directly comparable. Probably not trustworthy because execution times are very short.

To run benchmark:
```
cargo bench
```

Criterion output:

```
Enzyme:   1st order/forward  time:   [507.19 ps 507.52 ps 507.85 ps]
num-dual: 1st order/Dual64   time:   [311.51 ps 311.77 ps 312.03 ps]
```


