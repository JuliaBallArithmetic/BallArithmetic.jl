# API - NumericalTest

The rigour of the `Float32` and `Float64` paths rests on two properties of the machine the package
is running on: that every BLAS thread takes the rounding mode of its caller, and that the loops
Julia emits for a sum, a difference or a scalar multiple of arrays round as the mode says. Neither
is guaranteed, so both are tested when the package is loaded, and the outcome of each is recorded
in a flag that the product and sum routines read. Where a test failed the operations concerned are
formed entry by entry with the directed operations instead, or the product is enclosed without a
rounding mode at all.

## The rounding-mode tests

```@docs
BallArithmetic.NumericalTest.rounding_test
BallArithmetic.NumericalTest.array_rounding_test
```

## What the tests recorded

```@docs
BallArithmetic.NumericalTest.BLAS_ROUNDING_OK
BallArithmetic.NumericalTest.ARRAY_ROUNDING_OK
```
