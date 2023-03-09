
includet("test/testbench.jl")

dsv = testfun_create_cons(DensedSparseVector{Float64,UInt64}, 1_000_000); dsv1 = copy(dsv); dsv2 = similar(dsv); c = 2.0; f = *

DSV.HigherOrderFns._map_similar_zeropres!((x,y)->x-11*y, dsv2, dsv, dsv1)
DSV.HigherOrderFns.__map_similar_zeropres!((x,y)->x-11*y, dsv2, dsv, dsv1)
DSV.HigherOrderFns.__map_zeropres!((x,y)->x-11*y, dsv2, dsv, dsv1)

@benchmark DSV.HigherOrderFns._map_similar_zeropres!((x,y)->x-11*y, $dsv2, $dsv, $dsv1)
@benchmark DSV.HigherOrderFns.__map_similar_zeropres!((x,y)->x-11*y, $dsv2, $dsv, $dsv1)
@benchmark DSV.HigherOrderFns.__map_zeropres!((x,y)->x-11*y, $dsv2, $dsv, $dsv1)


dsv = testfun_create_dense(DensedSparseVector{Float64,UInt64}, 1_000_000, 800); dsv1 = copy(dsv); dsv2 = similar(dsv); c = 2.0; f = *


DSV.HigherOrderFns._map_similar_zeropres!((x,y)->x-11*y, dsv2, dsv, dsv1)
DSV.HigherOrderFns.__map_similar_zeropres!((x,y)->x-11*y, dsv2, dsv, dsv1)
DSV.HigherOrderFns.__map_zeropres!((x,y)->x-11*y, dsv2, dsv, dsv1)

@benchmark DSV.HigherOrderFns._map_similar_zeropres!((x,y)->x-11*y, $dsv2, $dsv, $dsv1)
@benchmark DSV.HigherOrderFns.__map_similar_zeropres!((x,y)->x-11*y, $dsv2, $dsv, $dsv1)
@benchmark DSV.HigherOrderFns.__map_zeropres!((x,y)->x-11*y, $dsv2, $dsv, $dsv1)


# Results
#=
julia> includet("test/testbench.jl")

julia> dsv = testfun_create_cons(DensedSparseVector{Float64,UInt64}, 1_000_000); dsv1 = copy(dsv); dsv2 = similar(dsv); c = 2.0; f = *
* (generic function with 397 methods)

julia> DSV.HigherOrderFns._map_similar_zeropres!((x,y)->x-11*y, dsv2, dsv, dsv1)
1000000-element DensedSparseVector{Float64, UInt64, Val{false}} with 900008 stored entries:
  [1      ]  =  -2.78145
  [2      ]  =  -7.34216
  [3      ]  =  -7.96613
  [4      ]  =  -7.25876
  [5      ]  =  -0.484581
  [6      ]  =  -8.27299
  [8      ]  =  -9.94363
  [9      ]  =  -4.28596
  [10     ]  =  -0.294222
  [11     ]  =  -4.37606
  [12     ]  =  -5.23205
  [13     ]  =  -9.5834
  [14     ]  =  -1.66221
  [15     ]  =  -7.12599
  [16     ]  =  -1.42482
             ⋮
  [999980 ]  =  -6.46084
  [999982 ]  =  -6.55984
  [999983 ]  =  -2.72279
  [999986 ]  =  -1.49623
  [999987 ]  =  -2.14685
  [999988 ]  =  -4.79483
  [999989 ]  =  -6.98758
  [999990 ]  =  -6.95162
  [999991 ]  =  -0.617884
  [999992 ]  =  -4.70251
  [999994 ]  =  -6.90691
  [999995 ]  =  -4.91288
  [999996 ]  =  -1.65322
  [999997 ]  =  -2.8409
  [999999 ]  =  -6.17828
  [1000000]  =  -5.02062

julia> DSV.HigherOrderFns.__map_similar_zeropres!((x,y)->x-11*y, dsv2, dsv, dsv1)
1000000-element DensedSparseVector{Float64, UInt64, Val{false}} with 900008 stored entries:
  [1      ]  =  -2.78145
  [2      ]  =  -7.34216
  [3      ]  =  -7.96613
  [4      ]  =  -7.25876
  [5      ]  =  -0.484581
  [6      ]  =  -8.27299
  [8      ]  =  -9.94363
  [9      ]  =  -4.28596
  [10     ]  =  -0.294222
  [11     ]  =  -4.37606
  [12     ]  =  -5.23205
  [13     ]  =  -9.5834
  [14     ]  =  -1.66221
  [15     ]  =  -7.12599
  [16     ]  =  -1.42482
             ⋮
  [999980 ]  =  -6.46084
  [999982 ]  =  -6.55984
  [999983 ]  =  -2.72279
  [999986 ]  =  -1.49623
  [999987 ]  =  -2.14685
  [999988 ]  =  -4.79483
  [999989 ]  =  -6.98758
  [999990 ]  =  -6.95162
  [999991 ]  =  -0.617884
  [999992 ]  =  -4.70251
  [999994 ]  =  -6.90691
  [999995 ]  =  -4.91288
  [999996 ]  =  -1.65322
  [999997 ]  =  -2.8409
  [999999 ]  =  -6.17828
  [1000000]  =  -5.02062

julia> DSV.HigherOrderFns.__map_zeropres!((x,y)->x-11*y, dsv2, dsv, dsv1)
1000000-element DensedSparseVector{Float64, UInt64, Val{false}} with 900008 stored entries:
  [1      ]  =  -2.78145
  [2      ]  =  -7.34216
  [3      ]  =  -7.96613
  [4      ]  =  -7.25876
  [5      ]  =  -0.484581
  [6      ]  =  -8.27299
  [8      ]  =  -9.94363
  [9      ]  =  -4.28596
  [10     ]  =  -0.294222
  [11     ]  =  -4.37606
  [12     ]  =  -5.23205
  [13     ]  =  -9.5834
  [14     ]  =  -1.66221
  [15     ]  =  -7.12599
  [16     ]  =  -1.42482
             ⋮
  [999980 ]  =  -6.46084
  [999982 ]  =  -6.55984
  [999983 ]  =  -2.72279
  [999986 ]  =  -1.49623
  [999987 ]  =  -2.14685
  [999988 ]  =  -4.79483
  [999989 ]  =  -6.98758
  [999990 ]  =  -6.95162
  [999991 ]  =  -0.617884
  [999992 ]  =  -4.70251
  [999994 ]  =  -6.90691
  [999995 ]  =  -4.91288
  [999996 ]  =  -1.65322
  [999997 ]  =  -2.8409
  [999999 ]  =  -6.17828
  [1000000]  =  -5.02062

julia> @benchmark DSV.HigherOrderFns._map_similar_zeropres!((x,y)->x-11*y, dsv2, dsv, dsv1)
BenchmarkTools.Trial: 1280 samples with 1 evaluation.
 Range (min … max):  3.575 ms …   4.521 ms  ┊ GC (min … max): 0.00% … 0.00%
 Time  (median):     3.880 ms               ┊ GC (median):    0.00%
 Time  (mean ± σ):   3.887 ms ± 111.640 μs  ┊ GC (mean ± σ):  0.00% ± 0.00%

                    ▄▅█▆▅▆▃▄▂▃▃▆▆▄▂▅▂▂                         
  ▂▂▂▂▃▃▃▃▃▃▃▄▂▄▄▄▅▆██████████████████▆▅▅▃▅▅▃▄▃▃▃▂▃▃▃▃▁▃▂▂▂▁▂ ▄
  3.57 ms         Histogram: frequency by time        4.24 ms <

 Memory estimate: 0 bytes, allocs estimate: 0.

julia> @benchmark DSV.HigherOrderFns.__map_similar_zeropres!((x,y)->x-11*y, dsv2, dsv, dsv1)
BenchmarkTools.Trial: 1296 samples with 1 evaluation.
 Range (min … max):  3.515 ms …   8.286 ms  ┊ GC (min … max): 0.00% … 0.00%
 Time  (median):     3.830 ms               ┊ GC (median):    0.00%
 Time  (mean ± σ):   3.839 ms ± 184.539 μs  ┊ GC (mean ± σ):  0.00% ± 0.00%

              ▁ ▁ ▁▁▂▇▆█▇▃▇▆▅▇▄▅▄▁▁                            
  ▂▁▃▃▃▃▅▅▄▄▅▇█▇█████████████████████▆▆▄▆▄▂▃▃▃▃▂▂▂▁▁▃▂▂▁▁▁▁▁▂ ▄
  3.51 ms         Histogram: frequency by time        4.29 ms <

 Memory estimate: 0 bytes, allocs estimate: 0.

julia> @benchmark DSV.HigherOrderFns.__map_zeropres!((x,y)->x-11*y, dsv2, dsv, dsv1)
BenchmarkTools.Trial: 282 samples with 1 evaluation.
 Range (min … max):  16.850 ms …  20.833 ms  ┊ GC (min … max): 0.00% … 0.00%
 Time  (median):     17.701 ms               ┊ GC (median):    0.00%
 Time  (mean ± σ):   17.716 ms ± 335.878 μs  ┊ GC (mean ± σ):  0.00% ± 0.00%

                       ▇▆██▂▅▃▄▁                                
  ▃▁▁▃▃▃▁▃▃▁▃▃▄▃▄▅▄▅▇▇██████████▇▅▁▄▃▄▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▃▁▁▁▁▃ ▃
  16.8 ms         Histogram: frequency by time           19 ms <

 Memory estimate: 0 bytes, allocs estimate: 0.

julia> dsv = testfun_create_dense(DensedSparseVector{Float64,UInt64}, 1_000_000, 800); dsv1 = copy(dsv); dsv2 = similar(dsv); c = 2.0; f = *
* (generic function with 397 methods)

julia> DSV.HigherOrderFns._map_similar_zeropres!((x,y)->x-11*y, dsv2, dsv, dsv1)
1000000-element DensedSparseVector{Float64, UInt64, Val{false}} with 949758 stored entries:
  [1     ]  =  -5.49051
  [2     ]  =  -2.18587
  [3     ]  =  -8.94245
  [4     ]  =  -3.53112
  [5     ]  =  -3.94255
  [6     ]  =  -9.53125
  [7     ]  =  -7.95547
  [8     ]  =  -4.9425
  [9     ]  =  -7.48415
  [10    ]  =  -5.78232
  [11    ]  =  -7.27935
  [12    ]  =  -0.0744801
  [13    ]  =  -1.99377
  [14    ]  =  -4.39243
  [15    ]  =  -6.82533
            ⋮
  [999923]  =  -3.30082
  [999924]  =  -9.52007
  [999925]  =  -5.50947
  [999926]  =  -4.43215
  [999927]  =  -7.25067
  [999928]  =  -4.56455
  [999929]  =  -4.32082
  [999930]  =  -7.3855
  [999931]  =  -5.67359
  [999932]  =  -2.27215
  [999933]  =  -7.48175
  [999934]  =  -5.07572
  [999935]  =  -1.61652
  [999936]  =  -1.71436
  [999937]  =  -2.30374
  [999938]  =  -5.17274

julia> DSV.HigherOrderFns.__map_similar_zeropres!((x,y)->x-11*y, dsv2, dsv, dsv1)
1000000-element DensedSparseVector{Float64, UInt64, Val{false}} with 949758 stored entries:
  [1     ]  =  -5.49051
  [2     ]  =  -2.18587
  [3     ]  =  -8.94245
  [4     ]  =  -3.53112
  [5     ]  =  -3.94255
  [6     ]  =  -9.53125
  [7     ]  =  -7.95547
  [8     ]  =  -4.9425
  [9     ]  =  -7.48415
  [10    ]  =  -5.78232
  [11    ]  =  -7.27935
  [12    ]  =  -0.0744801
  [13    ]  =  -1.99377
  [14    ]  =  -4.39243
  [15    ]  =  -6.82533
            ⋮
  [999923]  =  -3.30082
  [999924]  =  -9.52007
  [999925]  =  -5.50947
  [999926]  =  -4.43215
  [999927]  =  -7.25067
  [999928]  =  -4.56455
  [999929]  =  -4.32082
  [999930]  =  -7.3855
  [999931]  =  -5.67359
  [999932]  =  -2.27215
  [999933]  =  -7.48175
  [999934]  =  -5.07572
  [999935]  =  -1.61652
  [999936]  =  -1.71436
  [999937]  =  -2.30374
  [999938]  =  -5.17274

julia> DSV.HigherOrderFns.__map_zeropres!((x,y)->x-11*y, dsv2, dsv, dsv1)
1000000-element DensedSparseVector{Float64, UInt64, Val{false}} with 949758 stored entries:
  [1     ]  =  -5.49051
  [2     ]  =  -2.18587
  [3     ]  =  -8.94245
  [4     ]  =  -3.53112
  [5     ]  =  -3.94255
  [6     ]  =  -9.53125
  [7     ]  =  -7.95547
  [8     ]  =  -4.9425
  [9     ]  =  -7.48415
  [10    ]  =  -5.78232
  [11    ]  =  -7.27935
  [12    ]  =  -0.0744801
  [13    ]  =  -1.99377
  [14    ]  =  -4.39243
  [15    ]  =  -6.82533
            ⋮
  [999923]  =  -3.30082
  [999924]  =  -9.52007
  [999925]  =  -5.50947
  [999926]  =  -4.43215
  [999927]  =  -7.25067
  [999928]  =  -4.56455
  [999929]  =  -4.32082
  [999930]  =  -7.3855
  [999931]  =  -5.67359
  [999932]  =  -2.27215
  [999933]  =  -7.48175
  [999934]  =  -5.07572
  [999935]  =  -1.61652
  [999936]  =  -1.71436
  [999937]  =  -2.30374
  [999938]  =  -5.17274

julia> @benchmark DSV.HigherOrderFns._map_similar_zeropres!((x,y)->x-11*y, dsv2, dsv, dsv1)
BenchmarkTools.Trial: 4771 samples with 1 evaluation.
 Range (min … max):  965.357 μs …  1.435 ms  ┊ GC (min … max): 0.00% … 0.00%
 Time  (median):       1.035 ms              ┊ GC (median):    0.00%
 Time  (mean ± σ):     1.038 ms ± 29.025 μs  ┊ GC (mean ± σ):  0.00% ± 0.00%

                   ▃▆▇▆█▅▃▂                                     
  ▂▂▃▃▃▃▃▃▃▃▃▃▃▃▄▄▇██████████▇▆▅▄▄▃▃▃▃▃▃▃▂▂▂▂▂▂▂▂▂▂▁▁▂▁▂▁▂▂▁▂▂ ▃
  965 μs          Histogram: frequency by time         1.16 ms <

 Memory estimate: 0 bytes, allocs estimate: 0.

julia> @benchmark DSV.HigherOrderFns.__map_similar_zeropres!((x,y)->x-11*y, dsv2, dsv, dsv1)
BenchmarkTools.Trial: 4780 samples with 1 evaluation.
 Range (min … max):  963.497 μs …  1.441 ms  ┊ GC (min … max): 0.00% … 0.00%
 Time  (median):       1.034 ms              ┊ GC (median):    0.00%
 Time  (mean ± σ):     1.035 ms ± 27.835 μs  ┊ GC (mean ± σ):  0.00% ± 0.00%

                      ▁▃██▄█▅▄▃▁                                
  ▁▁▂▂▂▂▂▂▃▃▃▂▃▃▂▂▃▃▃▅███████████▇▆▅▅▄▄▃▃▃▂▂▂▁▂▂▂▁▁▁▁▁▁▁▁▁▁▁▁▁ ▃
  963 μs          Histogram: frequency by time         1.13 ms <

 Memory estimate: 0 bytes, allocs estimate: 0.

julia> @benchmark DSV.HigherOrderFns.__map_zeropres!((x,y)->x-11*y, dsv2, dsv, dsv1)
BenchmarkTools.Trial: 4318 samples with 1 evaluation.
 Range (min … max):  1.066 ms …  1.565 ms  ┊ GC (min … max): 0.00% … 0.00%
 Time  (median):     1.148 ms              ┊ GC (median):    0.00%
 Time  (mean ± σ):   1.147 ms ± 26.266 μs  ┊ GC (mean ± σ):  0.00% ± 0.00%

                            ▂▅▅▇▇█▇▇▅▄▄▁▂ ▁                   
  ▂▁▂▃▃▃▃▃▃▃▃▄▄▄▃▄▄▃▄▃▃▃▃▅▇████████████████▆▇▆▅▄▄▃▄▃▃▃▃▃▂▂▂▂ ▄
  1.07 ms        Histogram: frequency by time        1.21 ms <

 Memory estimate: 0 bytes, allocs estimate: 0.

=#

