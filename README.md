# MiniTorch Module 3

<img src="https://minitorch.github.io/minitorch.svg" width="50%">

* Docs: https://minitorch.github.io/

* Overview: https://minitorch.github.io/module3.html


You will need to modify `tensor_functions.py` slightly in this assignment.

* Tests:

```
python run_tests.py
```

* Note:

Several of the tests for this assignment will only run if you are on a GPU machine and will not
run on github's test infrastructure. Please follow the instructions to setup up a colab machine
to run these tests.

This assignment requires the following files from the previous assignments. You can get these by running

```bash
python sync_previous_module.py previous-module-dir current-module-dir
```

The files that will be synced are:

        minitorch/tensor_data.py minitorch/tensor_functions.py minitorch/tensor_ops.py minitorch/operators.py minitorch/scalar.py minitorch/scalar_functions.py minitorch/module.py minitorch/autodiff.py minitorch/module.py project/run_manual.py project/run_scalar.py project/run_tensor.py minitorch/operators.py minitorch/module.py minitorch/autodiff.py minitorch/tensor.py minitorch/datasets.py minitorch/testing.py minitorch/optim.py

# Task 3.1 + 3.2 Diagnostics Output

Diagnostics state that "Parallel structure is already optimal." for map, zip, reduce, and matrix_multiply.

```
MAP

================================================================================
 Parallel Accelerator Optimizing:  Function tensor_map.<locals>._map,
/content/mod3-dchoiboi00/minitorch/fast_ops.py (164)
================================================================================


Parallel loop listing for  Function tensor_map.<locals>._map, /content/mod3-dchoiboi00/minitorch/fast_ops.py (164)
---------------------------------------------------------------------------------------------|loop #ID
    def _map(                                                                                |
        out: Storage,                                                                        |
        out_shape: Shape,                                                                    |
        out_strides: Strides,                                                                |
        in_storage: Storage,                                                                 |
        in_shape: Shape,                                                                     |
        in_strides: Strides,                                                                 |
    ) -> None:                                                                               |
        # Check if shapes + strides are aligned. If so, run function on storage directly.    |
        if np.array_equal(in_strides, out_strides) and np.array_equal(                       |
            in_shape, out_shape                                                              |
        ):                                                                                   |
            for i in prange(len(out)):-------------------------------------------------------| #0
                out[i] = fn(in_storage[i])                                                   |
                                                                                             |
        # Otherwise, calculate indices and run parallel loop.                                |
        else:                                                                                |
            for i in prange(len(out)):-------------------------------------------------------| #1
                out_index: Index = np.empty(MAX_DIMS, dtype=np.int32)                        |
                in_index: Index = np.empty(MAX_DIMS, dtype=np.int32)                         |
                to_index(i, out_shape, out_index)                                            |
                broadcast_index(out_index, out_shape, in_shape, in_index)                    |
                o = index_to_position(out_index, out_strides)                                |
                j = index_to_position(in_index, in_strides)                                  |
                out[o] = fn(in_storage[j])                                                   |
--------------------------------- Fusing loops ---------------------------------
Attempting fusion of parallel loops (combines loops with similar properties)...
Following the attempted fusion of parallel for-loops there are 2 parallel for-
loop(s) (originating from loops labelled: #0, #1).
--------------------------------------------------------------------------------
----------------------------- Before Optimisation ------------------------------
--------------------------------------------------------------------------------
------------------------------ After Optimisation ------------------------------
Parallel structure is already optimal.
--------------------------------------------------------------------------------
--------------------------------------------------------------------------------

---------------------------Loop invariant code motion---------------------------
Allocation hoisting:
The memory allocation derived from the instruction at
/content/mod3-dchoiboi00/minitorch/fast_ops.py (182) is hoisted out of the
parallel loop labelled #1 (it will be performed before the loop is executed and
reused inside the loop):
   Allocation:: out_index: Index = np.empty(MAX_DIMS, dtype=np.int32)
    - numpy.empty() is used for the allocation.
The memory allocation derived from the instruction at
/content/mod3-dchoiboi00/minitorch/fast_ops.py (183) is hoisted out of the
parallel loop labelled #1 (it will be performed before the loop is executed and
reused inside the loop):
   Allocation:: in_index: Index = np.empty(MAX_DIMS, dtype=np.int32)
    - numpy.empty() is used for the allocation.
None


ZIP

================================================================================
 Parallel Accelerator Optimizing:  Function tensor_zip.<locals>._zip,
/content/mod3-dchoiboi00/minitorch/fast_ops.py (216)
================================================================================


Parallel loop listing for  Function tensor_zip.<locals>._zip, /content/mod3-dchoiboi00/minitorch/fast_ops.py (216)
---------------------------------------------------------------------------------------------|loop #ID
    def _zip(                                                                                |
        out: Storage,                                                                        |
        out_shape: Shape,                                                                    |
        out_strides: Strides,                                                                |
        a_storage: Storage,                                                                  |
        a_shape: Shape,                                                                      |
        a_strides: Strides,                                                                  |
        b_storage: Storage,                                                                  |
        b_shape: Shape,                                                                      |
        b_strides: Strides,                                                                  |
    ) -> None:                                                                               |
        # Check if shapes + strides are aligned. If so, run function on storage directly.    |
        if (                                                                                 |
            np.array_equal(a_strides, b_strides)                                             |
            and np.array_equal(a_strides, out_strides)                                       |
            and np.array_equal(a_shape, b_shape)                                             |
            and np.array_equal(a_shape, out_shape)                                           |
        ):                                                                                   |
            for i in prange(len(out)):-------------------------------------------------------| #2
                out[i] = fn(a_storage[i], b_storage[i])                                      |
                                                                                             |
        # Otherwise, calculate indices and run parallel loop.                                |
        else:                                                                                |
            for i in prange(len(out)):-------------------------------------------------------| #3
                out_index: Index = np.empty(MAX_DIMS, dtype=np.int32)                        |
                a_index: Index = np.empty(MAX_DIMS, dtype=np.int32)                          |
                b_index: Index = np.empty(MAX_DIMS, dtype=np.int32)                          |
                                                                                             |
                to_index(i, out_shape, out_index)                                            |
                broadcast_index(out_index, out_shape, a_shape, a_index)                      |
                broadcast_index(out_index, out_shape, b_shape, b_index)                      |
                                                                                             |
                o = index_to_position(out_index, out_strides)                                |
                a_pos = index_to_position(a_index, a_strides)                                |
                b_pos = index_to_position(b_index, b_strides)                                |
                                                                                             |
                out[o] = fn(a_storage[a_pos], b_storage[b_pos])                              |
--------------------------------- Fusing loops ---------------------------------
Attempting fusion of parallel loops (combines loops with similar properties)...
Following the attempted fusion of parallel for-loops there are 2 parallel for-
loop(s) (originating from loops labelled: #2, #3).
--------------------------------------------------------------------------------
----------------------------- Before Optimisation ------------------------------
--------------------------------------------------------------------------------
------------------------------ After Optimisation ------------------------------
Parallel structure is already optimal.
--------------------------------------------------------------------------------
--------------------------------------------------------------------------------

---------------------------Loop invariant code motion---------------------------
Allocation hoisting:
The memory allocation derived from the instruction at
/content/mod3-dchoiboi00/minitorch/fast_ops.py (240) is hoisted out of the
parallel loop labelled #3 (it will be performed before the loop is executed and
reused inside the loop):
   Allocation:: out_index: Index = np.empty(MAX_DIMS, dtype=np.int32)
    - numpy.empty() is used for the allocation.
The memory allocation derived from the instruction at
/content/mod3-dchoiboi00/minitorch/fast_ops.py (241) is hoisted out of the
parallel loop labelled #3 (it will be performed before the loop is executed and
reused inside the loop):
   Allocation:: a_index: Index = np.empty(MAX_DIMS, dtype=np.int32)
    - numpy.empty() is used for the allocation.
The memory allocation derived from the instruction at
/content/mod3-dchoiboi00/minitorch/fast_ops.py (242) is hoisted out of the
parallel loop labelled #3 (it will be performed before the loop is executed and
reused inside the loop):
   Allocation:: b_index: Index = np.empty(MAX_DIMS, dtype=np.int32)
    - numpy.empty() is used for the allocation.
None


REDUCE

================================================================================
 Parallel Accelerator Optimizing:  Function tensor_reduce.<locals>._reduce,
/content/mod3-dchoiboi00/minitorch/fast_ops.py (278)
================================================================================


Parallel loop listing for  Function tensor_reduce.<locals>._reduce, /content/mod3-dchoiboi00/minitorch/fast_ops.py (278)
---------------------------------------------------------------------|loop #ID
    def _reduce(                                                     |
        out: Storage,                                                |
        out_shape: Shape,                                            |
        out_strides: Strides,                                        |
        a_storage: Storage,                                          |
        a_shape: Shape,                                              |
        a_strides: Strides,                                          |
        reduce_dim: int,                                             |
    ) -> None:                                                       |
        for i in prange(len(out)):-----------------------------------| #4
            out_index: Index = np.empty(MAX_DIMS, dtype=np.int32)    |
            reduce_size = a_shape[reduce_dim]                        |
            to_index(i, out_shape, out_index)                        |
            o = index_to_position(out_index, out_strides)            |
                                                                     |
            # reduce across the reduce_dim                           |
            j = index_to_position(out_index, a_strides)              |
            acc = out[o]                                             |
            step = a_strides[reduce_dim]                             |
            for _ in range(reduce_size):                             |
                acc = fn(acc, a_storage[j])                          |
                j += step                                            |
            out[o] = acc                                             |
--------------------------------- Fusing loops ---------------------------------
Attempting fusion of parallel loops (combines loops with similar properties)...
Following the attempted fusion of parallel for-loops there are 1 parallel for-
loop(s) (originating from loops labelled: #4).
--------------------------------------------------------------------------------
----------------------------- Before Optimisation ------------------------------
--------------------------------------------------------------------------------
------------------------------ After Optimisation ------------------------------
Parallel structure is already optimal.
--------------------------------------------------------------------------------
--------------------------------------------------------------------------------

---------------------------Loop invariant code motion---------------------------
Allocation hoisting:
The memory allocation derived from the instruction at
/content/mod3-dchoiboi00/minitorch/fast_ops.py (288) is hoisted out of the
parallel loop labelled #4 (it will be performed before the loop is executed and
reused inside the loop):
   Allocation:: out_index: Index = np.empty(MAX_DIMS, dtype=np.int32)
    - numpy.empty() is used for the allocation.
None


MATRIX MULTIPLY

================================================================================
 Parallel Accelerator Optimizing:  Function _tensor_matrix_multiply,
/content/mod3-dchoiboi00/minitorch/fast_ops.py (305)
================================================================================


Parallel loop listing for  Function _tensor_matrix_multiply, /content/mod3-dchoiboi00/minitorch/fast_ops.py (305)
------------------------------------------------------------------------------|loop #ID
def _tensor_matrix_multiply(                                                  |
    out: Storage,                                                             |
    out_shape: Shape,                                                         |
    out_strides: Strides,                                                     |
    a_storage: Storage,                                                       |
    a_shape: Shape,                                                           |
    a_strides: Strides,                                                       |
    b_storage: Storage,                                                       |
    b_shape: Shape,                                                           |
    b_strides: Strides,                                                       |
) -> None:                                                                    |
    """NUMBA tensor matrix multiply function.                                 |
                                                                              |
    Should work for any tensor shapes that broadcast as long as               |
                                                                              |
    ```                                                                       |
    assert a_shape[-1] == b_shape[-2]                                         |
    ```                                                                       |
                                                                              |
    Optimizations:                                                            |
                                                                              |
    * Outer loop in parallel                                                  |
    * No index buffers or function calls                                      |
    * Inner loop should have no global writes, 1 multiply.                    |
                                                                              |
                                                                              |
    Args:                                                                     |
    ----                                                                      |
        out (Storage): storage for `out` tensor                               |
        out_shape (Shape): shape for `out` tensor                             |
        out_strides (Strides): strides for `out` tensor                       |
        a_storage (Storage): storage for `a` tensor                           |
        a_shape (Shape): shape for `a` tensor                                 |
        a_strides (Strides): strides for `a` tensor                           |
        b_storage (Storage): storage for `b` tensor                           |
        b_shape (Shape): shape for `b` tensor                                 |
        b_strides (Strides): strides for `b` tensor                           |
                                                                              |
    Returns:                                                                  |
    -------                                                                   |
        None : Fills in `out`                                                 |
                                                                              |
    """                                                                       |
    # A[batch, row, k] @ B[batch, k, col] = C[batch, row, col]                |
    # print("Running fast ops matrix multiply")                               |
    assert (                                                                  |
        a_shape[-1] == b_shape[-2]                                            |
    ), "Shapes do not match for matrix mult."  # Matrix multiply condition    |
    a_batch_stride = a_strides[0] if a_shape[0] > 1 else 0                    |
    b_batch_stride = b_strides[0] if b_shape[0] > 1 else 0                    |
                                                                              |
    # Parallel outer loop                                                     |
    for batch in prange(out_shape[0]):----------------------------------------| #5
        for row in range(out_shape[-2]):                                      |
            for col in range(out_shape[-1]):                                  |
                a_pos = (                                                     |
                    batch * a_batch_stride + row * a_strides[-2]              |
                )  # starting a-pos, local var                                |
                b_pos = (                                                     |
                    batch * b_batch_stride + col * b_strides[-1]              |
                )  # starting b-pos, local var                                |
                                                                              |
                acc = 0.0                                                     |
                for _ in range(a_shape[-1]):                                  |
                    acc += (                                                  |
                        a_storage[a_pos] * b_storage[b_pos]                   |
                    )  # accumulate, one multiply                             |
                    a_pos += a_strides[-1]  # Move along row in A             |
                    b_pos += b_strides[-2]  # Move along col in B             |
                                                                              |
                out_pos = (                                                   |
                    batch * out_strides[0]                                    |
                    + row * out_strides[-2]                                   |
                    + col * out_strides[-1]                                   |
                )                                                             |
                out[out_pos] = acc                                            |
--------------------------------- Fusing loops ---------------------------------
Attempting fusion of parallel loops (combines loops with similar properties)...
Following the attempted fusion of parallel for-loops there are 1 parallel for-
loop(s) (originating from loops labelled: #5).
--------------------------------------------------------------------------------
----------------------------- Before Optimisation ------------------------------
--------------------------------------------------------------------------------
------------------------------ After Optimisation ------------------------------
Parallel structure is already optimal.
--------------------------------------------------------------------------------
--------------------------------------------------------------------------------

---------------------------Loop invariant code motion---------------------------
Allocation hoisting:
No allocation hoisting found
None
```

# Task 3.5: Training

I ran CPU and GPU training on the __Simple__, __Split__, and __Xor__ datasets with 100 hidden layers. For the bigger model, I ran 200 hidden layers on the __Split__ dataset.

Results are below.

## Simple: 100 Hidden layers

CPU: ```python project/run_fast_tensor.py --BACKEND cpu --HIDDEN 100 --DATASET simple --RATE 0.05```

Time per epoch: 0.113 seconds

```
Epoch    1, Loss 4.913798611049, Correct  42, Time 7.995 seconds
Epoch   10, Loss 2.075415314029, Correct  46, Time 0.090 seconds
Epoch   20, Loss 1.217141558670, Correct  47, Time 0.094 seconds
Epoch   30, Loss 1.832635074990, Correct  50, Time 0.098 seconds
Epoch   40, Loss 0.946038270946, Correct  50, Time 0.093 seconds
Epoch   50, Loss 2.187597669774, Correct  50, Time 0.094 seconds
Epoch   60, Loss 0.689152289527, Correct  50, Time 0.094 seconds
Epoch   70, Loss 0.212794952276, Correct  50, Time 0.102 seconds
Epoch   80, Loss 0.930139263518, Correct  50, Time 0.096 seconds
Epoch   90, Loss 0.657770384777, Correct  50, Time 0.096 seconds
Epoch  100, Loss 0.733286703523, Correct  50, Time 0.100 seconds
Epoch  110, Loss 0.612776747056, Correct  50, Time 0.093 seconds
Epoch  120, Loss 0.446375980293, Correct  50, Time 0.093 seconds
Epoch  130, Loss 0.346044094541, Correct  50, Time 0.093 seconds
Epoch  140, Loss 0.246599984188, Correct  50, Time 0.111 seconds
Epoch  150, Loss 0.315772623477, Correct  50, Time 0.421 seconds
Epoch  160, Loss 0.440640296942, Correct  50, Time 0.242 seconds
Epoch  170, Loss 0.000230592541, Correct  50, Time 0.095 seconds
Epoch  180, Loss 0.377085839711, Correct  50, Time 0.108 seconds
Epoch  190, Loss 0.328544918023, Correct  50, Time 0.109 seconds
Epoch  200, Loss 0.206389901220, Correct  50, Time 0.096 seconds
Epoch  210, Loss 0.005525392808, Correct  50, Time 0.125 seconds
Epoch  220, Loss 0.207850467831, Correct  50, Time 0.139 seconds
Epoch  230, Loss 0.108844355481, Correct  50, Time 0.097 seconds
Epoch  240, Loss 0.186591716417, Correct  50, Time 0.112 seconds
Epoch  250, Loss 0.204502103544, Correct  50, Time 0.108 seconds
Epoch  260, Loss 0.166262565576, Correct  50, Time 0.109 seconds
Epoch  270, Loss 0.168744030081, Correct  50, Time 0.095 seconds
Epoch  280, Loss 0.009100023262, Correct  50, Time 0.106 seconds
Epoch  290, Loss 0.028022193145, Correct  50, Time 0.120 seconds
Epoch  300, Loss 0.195651360923, Correct  50, Time 0.093 seconds
Epoch  310, Loss 0.232834657941, Correct  50, Time 0.106 seconds
Epoch  320, Loss 0.099087424427, Correct  50, Time 0.094 seconds
Epoch  330, Loss 0.099229697654, Correct  50, Time 0.112 seconds
Epoch  340, Loss 0.007969120442, Correct  50, Time 0.112 seconds
Epoch  350, Loss 0.049346063135, Correct  50, Time 0.118 seconds
Epoch  360, Loss 0.157424293552, Correct  50, Time 0.107 seconds
Epoch  370, Loss 0.032388953987, Correct  50, Time 0.113 seconds
Epoch  380, Loss 0.104890257004, Correct  50, Time 0.105 seconds
Epoch  390, Loss 0.102396732020, Correct  50, Time 0.102 seconds
Epoch  400, Loss 0.089695084758, Correct  50, Time 0.111 seconds
Epoch  410, Loss 0.056865787268, Correct  50, Time 0.098 seconds
Epoch  420, Loss 0.122603264691, Correct  50, Time 0.129 seconds
Epoch  430, Loss 0.059576378006, Correct  50, Time 0.097 seconds
Epoch  440, Loss 0.075761403453, Correct  50, Time 0.098 seconds
Epoch  450, Loss 0.129947552818, Correct  50, Time 0.105 seconds
Epoch  460, Loss 0.054878316613, Correct  50, Time 0.143 seconds
Epoch  470, Loss 0.044160158365, Correct  50, Time 0.111 seconds
Epoch  480, Loss 0.110655087410, Correct  50, Time 0.123 seconds
Epoch  490, Loss 0.058420445692, Correct  50, Time 0.302 seconds
Epoch  500, Loss 0.147479953553, Correct  50, Time 0.168 seconds

Average Time Per Epoch: 0.113 seconds
```

GPU: ```python project/run_fast_tensor.py --BACKEND gpu --HIDDEN 100 --DATASET simple --RATE 0.05```

Time per epoch: 1.612 seconds

```
Epoch   10, Loss 2.186927007246, Correct  49, Time 1.949 seconds
Epoch   20, Loss 0.860599716991, Correct  50, Time 1.500 seconds
Epoch   30, Loss 0.624430837141, Correct  50, Time 1.492 seconds
Epoch   40, Loss 2.024824298693, Correct  50, Time 1.847 seconds
Epoch   50, Loss 0.605146484062, Correct  50, Time 1.495 seconds
Epoch   60, Loss 0.266236203111, Correct  50, Time 1.484 seconds
Epoch   70, Loss 0.435664291441, Correct  50, Time 1.658 seconds
Epoch   80, Loss 0.592079767392, Correct  50, Time 1.531 seconds
Epoch   90, Loss 0.122094043940, Correct  50, Time 1.478 seconds
Epoch  100, Loss 0.329530320767, Correct  50, Time 1.552 seconds
Epoch  110, Loss 0.119807073330, Correct  50, Time 1.478 seconds
Epoch  120, Loss 0.040599619060, Correct  50, Time 1.467 seconds
Epoch  130, Loss 0.068694944401, Correct  50, Time 1.637 seconds
Epoch  140, Loss 0.157384310629, Correct  50, Time 1.489 seconds
Epoch  150, Loss 0.648751299711, Correct  50, Time 1.519 seconds
Epoch  160, Loss 0.646895507912, Correct  50, Time 1.479 seconds
Epoch  170, Loss 0.697466271771, Correct  50, Time 1.882 seconds
Epoch  180, Loss 0.011070415794, Correct  50, Time 1.489 seconds
Epoch  190, Loss 0.114018429778, Correct  50, Time 1.475 seconds
Epoch  200, Loss 0.572609340622, Correct  50, Time 2.172 seconds
Epoch  210, Loss 0.034076594293, Correct  50, Time 1.481 seconds
Epoch  220, Loss 0.466923201464, Correct  50, Time 1.522 seconds
Epoch  230, Loss 0.143889685758, Correct  50, Time 2.155 seconds
Epoch  240, Loss 0.321561652347, Correct  50, Time 1.498 seconds
Epoch  250, Loss 0.245159902941, Correct  50, Time 1.450 seconds
Epoch  260, Loss 0.560211942238, Correct  50, Time 1.759 seconds
Epoch  270, Loss 0.276070918605, Correct  50, Time 1.487 seconds
Epoch  280, Loss 0.563469824102, Correct  50, Time 1.482 seconds
Epoch  290, Loss 0.468508821621, Correct  50, Time 1.488 seconds
Epoch  300, Loss 0.546691580055, Correct  50, Time 1.539 seconds
Epoch  310, Loss 0.430962221706, Correct  50, Time 1.505 seconds
Epoch  320, Loss 0.003889588298, Correct  50, Time 1.481 seconds
Epoch  330, Loss 0.396585365494, Correct  50, Time 1.485 seconds
Epoch  340, Loss 0.006966433168, Correct  50, Time 1.463 seconds
Epoch  350, Loss 0.076404858063, Correct  50, Time 1.464 seconds
Epoch  360, Loss 0.002560075834, Correct  50, Time 1.562 seconds
Epoch  370, Loss 0.003538316918, Correct  50, Time 1.470 seconds
Epoch  380, Loss 0.207126558472, Correct  50, Time 1.492 seconds
Epoch  390, Loss 0.048123334530, Correct  50, Time 1.541 seconds
Epoch  400, Loss 0.072665333668, Correct  50, Time 1.523 seconds
Epoch  410, Loss 0.215751676207, Correct  50, Time 1.471 seconds
Epoch  420, Loss 0.378912151178, Correct  50, Time 1.549 seconds
Epoch  430, Loss 0.202863293495, Correct  50, Time 1.464 seconds
Epoch  440, Loss 0.002079004058, Correct  50, Time 1.488 seconds
Epoch  450, Loss 0.399519889211, Correct  50, Time 1.566 seconds
Epoch  460, Loss 0.490940544030, Correct  50, Time 1.478 seconds
Epoch  470, Loss 0.488971195417, Correct  50, Time 1.485 seconds
Epoch  480, Loss 0.404785514507, Correct  50, Time 1.749 seconds
Epoch  490, Loss 0.186246795507, Correct  50, Time 1.547 seconds
Epoch  500, Loss 0.018064503616, Correct  50, Time 1.554 seconds

Average Time Per Epoch: 1.612 seconds
```

## Split: 100 Hidden layers

CPU: ```python project/run_fast_tensor.py --BACKEND cpu --HIDDEN 100 --DATASET split --RATE 0.05```

Time per epoch: 0.114 seconds

```
Epoch    1, Loss 9.568842434823, Correct  32, Time 8.120 seconds
Epoch   10, Loss 5.354931910489, Correct  38, Time 0.094 seconds
Epoch   20, Loss 4.416900621041, Correct  39, Time 0.094 seconds
Epoch   30, Loss 4.575975453644, Correct  47, Time 0.100 seconds
Epoch   40, Loss 2.506596689080, Correct  46, Time 0.093 seconds
Epoch   50, Loss 2.781091070170, Correct  49, Time 0.094 seconds
Epoch   60, Loss 2.125449073682, Correct  49, Time 0.104 seconds
Epoch   70, Loss 1.441830467223, Correct  50, Time 0.100 seconds
Epoch   80, Loss 1.570551578639, Correct  49, Time 0.090 seconds
Epoch   90, Loss 1.120282131093, Correct  50, Time 0.093 seconds
Epoch  100, Loss 1.737302928604, Correct  50, Time 0.135 seconds
Epoch  110, Loss 0.302956292801, Correct  50, Time 0.132 seconds
Epoch  120, Loss 0.867672999526, Correct  50, Time 0.137 seconds
Epoch  130, Loss 1.111432402638, Correct  50, Time 0.093 seconds
Epoch  140, Loss 0.505753696374, Correct  50, Time 0.093 seconds
Epoch  150, Loss 1.919045351179, Correct  48, Time 0.096 seconds
Epoch  160, Loss 0.733246199883, Correct  50, Time 0.093 seconds
Epoch  170, Loss 0.633538558694, Correct  50, Time 0.156 seconds
Epoch  180, Loss 0.795263622948, Correct  50, Time 0.126 seconds
Epoch  190, Loss 0.619231169746, Correct  50, Time 0.154 seconds
Epoch  200, Loss 0.726668548583, Correct  50, Time 0.179 seconds
Epoch  210, Loss 0.353994823267, Correct  50, Time 0.132 seconds
Epoch  220, Loss 0.436291479421, Correct  50, Time 0.166 seconds
Epoch  230, Loss 0.504282647412, Correct  50, Time 0.171 seconds
Epoch  240, Loss 0.286144635704, Correct  50, Time 0.157 seconds
Epoch  250, Loss 0.244699541455, Correct  50, Time 0.131 seconds
Epoch  260, Loss 0.280077070172, Correct  50, Time 0.126 seconds
Epoch  270, Loss 0.099463384688, Correct  50, Time 0.116 seconds
Epoch  280, Loss 0.168860067166, Correct  50, Time 0.093 seconds
Epoch  290, Loss 0.309050633460, Correct  50, Time 0.093 seconds
Epoch  300, Loss 0.085610889624, Correct  50, Time 0.096 seconds
Epoch  310, Loss 0.281907763425, Correct  50, Time 0.094 seconds
Epoch  320, Loss 0.159616736715, Correct  50, Time 0.094 seconds
Epoch  330, Loss 0.075456794905, Correct  50, Time 0.093 seconds
Epoch  340, Loss 0.196041723122, Correct  50, Time 0.105 seconds
Epoch  350, Loss 0.242566555934, Correct  50, Time 0.093 seconds
Epoch  360, Loss 0.036181654968, Correct  50, Time 0.094 seconds
Epoch  370, Loss 0.071251067982, Correct  50, Time 0.094 seconds
Epoch  380, Loss 0.156559802643, Correct  50, Time 0.093 seconds
Epoch  390, Loss 0.053586476791, Correct  50, Time 0.093 seconds
Epoch  400, Loss 0.088492422147, Correct  50, Time 0.093 seconds
Epoch  410, Loss 0.141802377957, Correct  50, Time 0.093 seconds
Epoch  420, Loss 0.143469306921, Correct  50, Time 0.093 seconds
Epoch  430, Loss 0.187799807003, Correct  50, Time 0.094 seconds
Epoch  440, Loss 0.187050289791, Correct  50, Time 0.110 seconds
Epoch  450, Loss 0.023277253410, Correct  50, Time 0.093 seconds
Epoch  460, Loss 0.049358003491, Correct  50, Time 0.092 seconds
Epoch  470, Loss 0.087617351439, Correct  50, Time 0.094 seconds
Epoch  480, Loss 0.095760272183, Correct  50, Time 0.097 seconds
Epoch  490, Loss 0.073988918372, Correct  50, Time 0.097 seconds
Epoch  500, Loss 0.135510296373, Correct  50, Time 0.107 seconds

Average Time Per Epoch: 0.114 seconds
```

GPU: ```python project/run_fast_tensor.py --BACKEND gpu --HIDDEN 100 --DATASET split --RATE 0.05```

Time per epoch: 1.635 seconds

```
Epoch   10, Loss 6.557502791172, Correct  44, Time 1.583 seconds
Epoch   20, Loss 3.546942244248, Correct  43, Time 1.792 seconds
Epoch   30, Loss 2.438184476820, Correct  43, Time 1.486 seconds
Epoch   40, Loss 3.594443962601, Correct  45, Time 1.601 seconds
Epoch   50, Loss 3.087166606020, Correct  47, Time 1.572 seconds
Epoch   60, Loss 2.536739133174, Correct  45, Time 1.484 seconds
Epoch   70, Loss 1.457035205044, Correct  49, Time 1.547 seconds
Epoch   80, Loss 1.568875159590, Correct  49, Time 1.513 seconds
Epoch   90, Loss 2.916868655837, Correct  49, Time 1.595 seconds
Epoch  100, Loss 2.275756361847, Correct  49, Time 1.559 seconds
Epoch  110, Loss 1.058941247951, Correct  49, Time 1.511 seconds
Epoch  120, Loss 1.521393655150, Correct  49, Time 1.490 seconds
Epoch  130, Loss 1.407866020611, Correct  49, Time 1.655 seconds
Epoch  140, Loss 0.447245127592, Correct  49, Time 1.509 seconds
Epoch  150, Loss 0.486339932889, Correct  49, Time 1.478 seconds
Epoch  160, Loss 1.725186085162, Correct  50, Time 1.513 seconds
Epoch  170, Loss 1.839438698119, Correct  50, Time 1.506 seconds
Epoch  180, Loss 0.476791439830, Correct  49, Time 1.529 seconds
Epoch  190, Loss 0.958155854363, Correct  49, Time 1.533 seconds
Epoch  200, Loss 0.513076717244, Correct  49, Time 1.684 seconds
Epoch  210, Loss 0.993279783584, Correct  49, Time 1.505 seconds
Epoch  220, Loss 0.201173854579, Correct  50, Time 1.492 seconds
Epoch  230, Loss 0.277888685716, Correct  49, Time 1.561 seconds
Epoch  240, Loss 0.151350213525, Correct  50, Time 1.500 seconds
Epoch  250, Loss 0.265646073405, Correct  50, Time 1.521 seconds
Epoch  260, Loss 0.369796958987, Correct  50, Time 1.539 seconds
Epoch  270, Loss 0.455961938049, Correct  50, Time 1.539 seconds
Epoch  280, Loss 0.441939484245, Correct  49, Time 1.505 seconds
Epoch  290, Loss 0.178701154695, Correct  50, Time 1.504 seconds
Epoch  300, Loss 0.202000546196, Correct  50, Time 1.789 seconds
Epoch  310, Loss 0.256407361720, Correct  50, Time 1.501 seconds
Epoch  320, Loss 0.121705221656, Correct  50, Time 1.476 seconds
Epoch  330, Loss 0.081159035805, Correct  50, Time 2.100 seconds
Epoch  340, Loss 0.656301270345, Correct  50, Time 1.512 seconds
Epoch  350, Loss 0.217132970701, Correct  50, Time 1.504 seconds
Epoch  360, Loss 0.030371206826, Correct  50, Time 2.147 seconds
Epoch  370, Loss 0.094420608653, Correct  50, Time 1.497 seconds
Epoch  380, Loss 0.649109852558, Correct  50, Time 1.460 seconds
Epoch  390, Loss 0.119176272489, Correct  50, Time 2.328 seconds
Epoch  400, Loss 0.168715800402, Correct  50, Time 1.479 seconds
Epoch  410, Loss 0.074739833294, Correct  50, Time 1.470 seconds
Epoch  420, Loss 0.674661168091, Correct  50, Time 1.853 seconds
Epoch  430, Loss 0.158325398797, Correct  50, Time 1.478 seconds
Epoch  440, Loss 0.615297781113, Correct  50, Time 1.495 seconds
Epoch  450, Loss 0.041049553656, Correct  50, Time 1.520 seconds
Epoch  460, Loss 0.161266435293, Correct  50, Time 1.532 seconds
Epoch  470, Loss 0.083421971440, Correct  50, Time 1.482 seconds
Epoch  480, Loss 0.446506631770, Correct  50, Time 1.537 seconds
Epoch  490, Loss 0.489232663166, Correct  50, Time 1.564 seconds
Epoch  500, Loss 0.438190418675, Correct  50, Time 1.528 seconds

Average Time Per Epoch: 1.635 seconds
```

## Xor: 100 Hidden layers

CPU: ```python project/run_fast_tensor.py --BACKEND cpu --HIDDEN 100 --DATASET xor --RATE 0.1```

Time per epoch: 0.096 seconds

```
Epoch    1, Loss 9.874909575953, Correct  24, Time 7.915 seconds
Epoch   10, Loss 5.144730465077, Correct  42, Time 0.199 seconds
Epoch   20, Loss 3.377386416987, Correct  43, Time 0.094 seconds
Epoch   30, Loss 2.859276128667, Correct  45, Time 0.095 seconds
Epoch   40, Loss 2.720278146493, Correct  46, Time 0.093 seconds
Epoch   50, Loss 2.898676268286, Correct  44, Time 0.092 seconds
Epoch   60, Loss 2.384455602632, Correct  49, Time 0.098 seconds
Epoch   70, Loss 1.703351960982, Correct  49, Time 0.093 seconds
Epoch   80, Loss 2.296876204684, Correct  50, Time 0.094 seconds
Epoch   90, Loss 2.106967322437, Correct  50, Time 0.093 seconds
Epoch  100, Loss 1.134021101380, Correct  50, Time 0.092 seconds
Epoch  110, Loss 0.827902702509, Correct  50, Time 0.137 seconds
Epoch  120, Loss 1.133468960759, Correct  50, Time 0.093 seconds
Epoch  130, Loss 0.093208444389, Correct  50, Time 0.094 seconds
Epoch  140, Loss 0.528798708659, Correct  50, Time 0.099 seconds
Epoch  150, Loss 1.245328806613, Correct  50, Time 0.093 seconds
Epoch  160, Loss 0.589215141987, Correct  50, Time 0.097 seconds
Epoch  170, Loss 0.258704823794, Correct  50, Time 0.093 seconds
Epoch  180, Loss 0.571530356332, Correct  50, Time 0.093 seconds
Epoch  190, Loss 0.868345959968, Correct  50, Time 0.093 seconds
Epoch  200, Loss 0.293168396672, Correct  50, Time 0.095 seconds
Epoch  210, Loss 0.282311649145, Correct  49, Time 0.093 seconds
Epoch  220, Loss 0.419091103950, Correct  50, Time 0.092 seconds
Epoch  230, Loss 0.227699913128, Correct  50, Time 0.093 seconds
Epoch  240, Loss 0.885862285565, Correct  50, Time 0.093 seconds
Epoch  250, Loss 0.922188020714, Correct  49, Time 0.093 seconds
Epoch  260, Loss 0.499290747039, Correct  50, Time 0.093 seconds
Epoch  270, Loss 0.108713891564, Correct  50, Time 0.092 seconds
Epoch  280, Loss 0.113512080136, Correct  50, Time 0.093 seconds
Epoch  290, Loss 0.277616236780, Correct  50, Time 0.088 seconds
Epoch  300, Loss 0.217286497674, Correct  50, Time 0.093 seconds
Epoch  310, Loss 0.031311622370, Correct  50, Time 0.093 seconds
Epoch  320, Loss 0.173044310836, Correct  50, Time 0.093 seconds
Epoch  330, Loss 0.420542650353, Correct  50, Time 0.093 seconds
Epoch  340, Loss 0.334329943437, Correct  50, Time 0.093 seconds
Epoch  350, Loss 0.209237839866, Correct  50, Time 0.092 seconds
Epoch  360, Loss 0.090698679734, Correct  50, Time 0.127 seconds
Epoch  370, Loss 0.275459730502, Correct  50, Time 0.093 seconds
Epoch  380, Loss 0.099347346991, Correct  50, Time 0.092 seconds
Epoch  390, Loss 0.092788072991, Correct  50, Time 0.092 seconds
Epoch  400, Loss 0.099631841192, Correct  50, Time 0.094 seconds
Epoch  410, Loss 0.020580493864, Correct  50, Time 0.093 seconds
Epoch  420, Loss 0.139374144321, Correct  50, Time 0.093 seconds
Epoch  430, Loss 0.076938019846, Correct  50, Time 0.093 seconds
Epoch  440, Loss 0.070123715312, Correct  50, Time 0.093 seconds
Epoch  450, Loss 0.166714723037, Correct  50, Time 0.092 seconds
Epoch  460, Loss 0.016668665913, Correct  50, Time 0.093 seconds
Epoch  470, Loss 0.228242425444, Correct  50, Time 0.092 seconds
Epoch  480, Loss 0.047983820193, Correct  50, Time 0.092 seconds
Epoch  490, Loss 0.015913771029, Correct  50, Time 0.093 seconds
Epoch  500, Loss 0.068930332946, Correct  50, Time 0.093 seconds

Average Time Per Epoch: 0.096 seconds
```

GPU: ```python project/run_fast_tensor.py --BACKEND gpu --HIDDEN 100 --DATASET xor --RATE 0.05```

Time per epoch: 1.614 seconds

```
Epoch   10, Loss 3.072142927386, Correct  41, Time 1.522 seconds
Epoch   20, Loss 4.047455999541, Correct  44, Time 2.326 seconds
Epoch   30, Loss 4.408904414713, Correct  45, Time 1.478 seconds
Epoch   40, Loss 2.924953475851, Correct  45, Time 1.529 seconds
Epoch   50, Loss 3.324396408169, Correct  44, Time 1.507 seconds
Epoch   60, Loss 2.458711430250, Correct  46, Time 1.477 seconds
Epoch   70, Loss 3.658043775590, Correct  46, Time 1.556 seconds
Epoch   80, Loss 3.650452089410, Correct  47, Time 1.915 seconds
Epoch   90, Loss 1.059280139059, Correct  48, Time 1.485 seconds
Epoch  100, Loss 0.512491759589, Correct  46, Time 1.600 seconds
Epoch  110, Loss 2.585739746762, Correct  47, Time 2.226 seconds
Epoch  120, Loss 0.393709204754, Correct  48, Time 1.495 seconds
Epoch  130, Loss 0.686998858791, Correct  48, Time 1.549 seconds
Epoch  140, Loss 0.562762956565, Correct  47, Time 1.974 seconds
Epoch  150, Loss 2.328265131822, Correct  48, Time 1.526 seconds
Epoch  160, Loss 1.045544677476, Correct  48, Time 1.532 seconds
Epoch  170, Loss 1.795325385380, Correct  47, Time 1.753 seconds
Epoch  180, Loss 2.035970986753, Correct  48, Time 1.511 seconds
Epoch  190, Loss 1.700338064782, Correct  48, Time 1.510 seconds
Epoch  200, Loss 0.512391601159, Correct  49, Time 1.549 seconds
Epoch  210, Loss 0.515225329026, Correct  49, Time 1.487 seconds
Epoch  220, Loss 1.000411777661, Correct  49, Time 1.460 seconds
Epoch  230, Loss 1.496701397332, Correct  49, Time 1.550 seconds
Epoch  240, Loss 1.476681062099, Correct  49, Time 1.496 seconds
Epoch  250, Loss 1.702416264014, Correct  50, Time 1.482 seconds
Epoch  260, Loss 1.689577982661, Correct  50, Time 1.516 seconds
Epoch  270, Loss 1.440419072846, Correct  49, Time 1.473 seconds
Epoch  280, Loss 2.482106107376, Correct  50, Time 1.509 seconds
Epoch  290, Loss 0.427555947487, Correct  49, Time 1.493 seconds
Epoch  300, Loss 0.135523291221, Correct  49, Time 1.778 seconds
Epoch  310, Loss 1.292302411743, Correct  50, Time 1.501 seconds
Epoch  320, Loss 1.990939915365, Correct  50, Time 1.476 seconds
Epoch  330, Loss 0.649530105899, Correct  49, Time 1.923 seconds
Epoch  340, Loss 1.312887684707, Correct  50, Time 1.510 seconds
Epoch  350, Loss 0.051982027193, Correct  49, Time 1.484 seconds
Epoch  360, Loss 0.850168508553, Correct  49, Time 2.020 seconds
Epoch  370, Loss 0.980745379677, Correct  50, Time 1.467 seconds
Epoch  380, Loss 1.322075567828, Correct  50, Time 1.485 seconds
Epoch  390, Loss 0.548857409635, Correct  50, Time 2.165 seconds
Epoch  400, Loss 1.443126157288, Correct  50, Time 1.501 seconds
Epoch  410, Loss 0.413075957886, Correct  50, Time 1.501 seconds
Epoch  420, Loss 0.616951536348, Correct  49, Time 2.266 seconds
Epoch  430, Loss 0.316681101161, Correct  50, Time 1.467 seconds
Epoch  440, Loss 1.104139711895, Correct  50, Time 1.512 seconds
Epoch  450, Loss 0.042561222393, Correct  50, Time 1.988 seconds
Epoch  460, Loss 0.165440533097, Correct  50, Time 1.470 seconds
Epoch  470, Loss 1.187276684659, Correct  50, Time 1.485 seconds
Epoch  480, Loss 0.721545869172, Correct  50, Time 1.676 seconds
Epoch  490, Loss 1.080477677253, Correct  50, Time 1.560 seconds
Epoch  500, Loss 0.476336318504, Correct  50, Time 1.463 seconds

Average Time Per Epoch: 1.614 seconds
```

## Bigger Model - Split: 200 Hidden layers

CPU: ```python project/run_fast_tensor.py --BACKEND cpu --HIDDEN 200 --DATASET split --RATE 0.05```

Time per epoch: 0.254 seconds

```
Epoch    1, Loss 14.308816807878, Correct  19, Time 11.815 seconds
Epoch   10, Loss 5.352658492707, Correct  43, Time 0.443 seconds
Epoch   20, Loss 5.051928400066, Correct  45, Time 0.173 seconds
Epoch   30, Loss 2.639829901906, Correct  47, Time 0.181 seconds
Epoch   40, Loss 3.356547503081, Correct  48, Time 0.601 seconds
Epoch   50, Loss 2.177518871575, Correct  48, Time 0.299 seconds
Epoch   60, Loss 2.474113187382, Correct  45, Time 0.189 seconds
Epoch   70, Loss 3.274793079614, Correct  45, Time 0.209 seconds
Epoch   80, Loss 1.599735171351, Correct  44, Time 0.209 seconds
Epoch   90, Loss 1.082503455835, Correct  46, Time 0.207 seconds
Epoch  100, Loss 1.754014230305, Correct  49, Time 0.247 seconds
Epoch  110, Loss 1.679482655968, Correct  50, Time 0.199 seconds
Epoch  120, Loss 2.026880863350, Correct  45, Time 0.201 seconds
Epoch  130, Loss 0.574404821023, Correct  47, Time 0.207 seconds
Epoch  140, Loss 0.851574648607, Correct  45, Time 0.201 seconds
Epoch  150, Loss 0.939627813462, Correct  50, Time 1.304 seconds
Epoch  160, Loss 0.970770167145, Correct  49, Time 0.350 seconds
Epoch  170, Loss 0.343787977554, Correct  50, Time 0.257 seconds
Epoch  180, Loss 1.002695886259, Correct  49, Time 0.292 seconds
Epoch  190, Loss 0.353034574825, Correct  50, Time 0.270 seconds
Epoch  200, Loss 0.250520323739, Correct  49, Time 0.189 seconds
Epoch  210, Loss 0.951980330105, Correct  50, Time 0.205 seconds
Epoch  220, Loss 0.356989748803, Correct  48, Time 0.257 seconds
Epoch  230, Loss 0.345501327356, Correct  50, Time 0.173 seconds
Epoch  240, Loss 0.862305468638, Correct  50, Time 0.172 seconds
Epoch  250, Loss 0.546949653090, Correct  48, Time 0.169 seconds
Epoch  260, Loss 0.784506367980, Correct  50, Time 0.170 seconds
Epoch  270, Loss 0.056895031242, Correct  47, Time 0.173 seconds
Epoch  280, Loss 0.379469492676, Correct  50, Time 0.177 seconds
Epoch  290, Loss 1.041650888159, Correct  50, Time 0.170 seconds
Epoch  300, Loss 0.062452645427, Correct  48, Time 0.177 seconds
Epoch  310, Loss 1.214136123295, Correct  50, Time 0.199 seconds
Epoch  320, Loss 0.990169537217, Correct  50, Time 0.232 seconds
Epoch  330, Loss 1.079087735707, Correct  49, Time 0.222 seconds
Epoch  340, Loss 0.326318733384, Correct  50, Time 0.374 seconds
Epoch  350, Loss 0.524172714928, Correct  50, Time 0.232 seconds
Epoch  360, Loss 1.071805903829, Correct  48, Time 0.169 seconds
Epoch  370, Loss 1.379612063015, Correct  50, Time 0.175 seconds
Epoch  380, Loss 0.132074314353, Correct  49, Time 0.187 seconds
Epoch  390, Loss 0.234638155720, Correct  50, Time 0.185 seconds
Epoch  400, Loss 0.365711648282, Correct  50, Time 0.170 seconds
Epoch  410, Loss 0.689941503994, Correct  50, Time 0.173 seconds
Epoch  420, Loss 0.454497694881, Correct  50, Time 0.169 seconds
Epoch  430, Loss 0.157926241258, Correct  49, Time 0.170 seconds
Epoch  440, Loss 0.712829414282, Correct  50, Time 0.169 seconds
Epoch  450, Loss 0.424353417198, Correct  50, Time 0.174 seconds
Epoch  460, Loss 0.894024208986, Correct  50, Time 0.203 seconds
Epoch  470, Loss 0.762280134798, Correct  49, Time 0.178 seconds
Epoch  480, Loss 0.064897915776, Correct  50, Time 0.197 seconds
Epoch  490, Loss 0.947209021183, Correct  50, Time 0.177 seconds
Epoch  500, Loss 0.556277921370, Correct  50, Time 0.172 seconds

Average Time Per Epoch: 0.254 seconds
```

GPU: ```python project/run_fast_tensor.py --BACKEND gpu --HIDDEN 200 --DATASET split --RATE 0.05```

Time per epoch: 1.683 seconds

```
Epoch   10, Loss 3.177207288037, Correct  44, Time 1.619 seconds
Epoch   20, Loss 1.870884951819, Correct  45, Time 1.562 seconds
Epoch   30, Loss 1.785581365226, Correct  49, Time 1.601 seconds
Epoch   40, Loss 2.409042836319, Correct  48, Time 1.607 seconds
Epoch   50, Loss 2.217048610524, Correct  48, Time 2.266 seconds
Epoch   60, Loss 1.801228640415, Correct  48, Time 1.547 seconds
Epoch   70, Loss 2.285450200941, Correct  49, Time 1.616 seconds
Epoch   80, Loss 0.752694786983, Correct  49, Time 1.554 seconds
Epoch   90, Loss 1.527938695888, Correct  49, Time 1.555 seconds
Epoch  100, Loss 0.343745300782, Correct  49, Time 2.084 seconds
Epoch  110, Loss 0.389617375216, Correct  48, Time 1.583 seconds
Epoch  120, Loss 0.889142843964, Correct  49, Time 1.548 seconds
Epoch  130, Loss 0.881434334852, Correct  49, Time 1.807 seconds
Epoch  140, Loss 1.035695978412, Correct  49, Time 1.549 seconds
Epoch  150, Loss 0.267557546822, Correct  49, Time 1.664 seconds
Epoch  160, Loss 0.499868531416, Correct  50, Time 1.561 seconds
Epoch  170, Loss 2.193375600412, Correct  48, Time 1.570 seconds
Epoch  180, Loss 0.276939201743, Correct  49, Time 2.087 seconds
Epoch  190, Loss 0.819657538782, Correct  49, Time 1.565 seconds
Epoch  200, Loss 0.195175403353, Correct  49, Time 1.647 seconds
Epoch  210, Loss 1.115159200300, Correct  50, Time 1.541 seconds
Epoch  220, Loss 1.009731020125, Correct  49, Time 1.572 seconds
Epoch  230, Loss 0.354020671690, Correct  50, Time 2.349 seconds
Epoch  240, Loss 0.041321847344, Correct  50, Time 1.891 seconds
Epoch  250, Loss 0.520667656810, Correct  49, Time 1.549 seconds
Epoch  260, Loss 0.745142182695, Correct  50, Time 1.647 seconds
Epoch  270, Loss 0.681849953274, Correct  49, Time 1.586 seconds
Epoch  280, Loss 0.096297790005, Correct  50, Time 2.283 seconds
Epoch  290, Loss 0.136703273813, Correct  50, Time 1.553 seconds
Epoch  300, Loss 0.093902960237, Correct  50, Time 1.555 seconds
Epoch  310, Loss 0.696392310611, Correct  49, Time 1.562 seconds
Epoch  320, Loss 0.028075435162, Correct  49, Time 1.564 seconds
Epoch  330, Loss 0.045372229999, Correct  50, Time 1.917 seconds
Epoch  340, Loss 0.579211491623, Correct  50, Time 1.585 seconds
Epoch  350, Loss 1.191656928923, Correct  50, Time 1.567 seconds
Epoch  360, Loss 0.023954819887, Correct  50, Time 2.024 seconds
Epoch  370, Loss 0.635668616072, Correct  50, Time 1.537 seconds
Epoch  380, Loss 0.057341222310, Correct  50, Time 1.546 seconds
Epoch  390, Loss 0.009814492487, Correct  50, Time 1.592 seconds
Epoch  400, Loss 1.232572042118, Correct  50, Time 1.540 seconds
Epoch  410, Loss 0.072029700735, Correct  50, Time 2.178 seconds
Epoch  420, Loss 0.025766547103, Correct  50, Time 1.623 seconds
Epoch  430, Loss 0.022684668298, Correct  48, Time 1.552 seconds
Epoch  440, Loss 0.585145126855, Correct  50, Time 1.534 seconds
Epoch  450, Loss 0.321902805196, Correct  50, Time 1.566 seconds
Epoch  460, Loss 0.662359322356, Correct  50, Time 2.095 seconds
Epoch  470, Loss 0.348441888978, Correct  50, Time 1.543 seconds
Epoch  480, Loss 0.035606529437, Correct  50, Time 1.555 seconds
Epoch  490, Loss 0.446257463675, Correct  50, Time 1.860 seconds
Epoch  500, Loss 0.025331063293, Correct  50, Time 1.543 seconds

Average Time Per Epoch: 1.683 seconds
```