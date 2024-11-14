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

I ran CPU and GPU training on the __Simple__, __Split__, and __Xor__ datasets. For the bigger model, I ran 100 hidden layers on the __Split__ dataset.

Results are below.


