import jax

# more test cases: https://github.com/google/jax/blob/main/tests/api_test.py


def test_jit_lower_add():
    def f(x, y):
        return x + y

    lowered = jax.jit(f).lower(1.0, 2.0)
    print(jax.jit(f).as_text())
    hlo_ir = lowered.as_text(dialect="hlo")
    mhlo_ir = lowered.as_text(dialect="mhlo")
    stablehlo_ir = lowered.as_text(dialect="stablehlo")
    print(
        "hlo_ir: \n{}\nmhlo: \n{}\nstablehlo: \n{}".format(
            hlo_ir, mhlo_ir, stablehlo_ir
        )
    )
    mhlo_str = str(lowered.compiler_ir("mhlo"))
    stablehlo_str = str(lowered.compiler_ir("stablehlo"))
    print("mhlo_str:\n{}\nstablehlo_str:\n{}".format(mhlo_str, stablehlo_str))


if __name__ == "__main__":
    test_jit_lower_add()
