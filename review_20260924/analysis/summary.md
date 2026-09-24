# KV AllGather ablation

These are source-bound MHA operator measurements, not full-model results. All arms use the same inputs per case and rank. The attention backend is forced PyTorch Flash SDPA, with BF16, 32 Q/K/V heads, head dimension 128, and noncausal attention.

Each timing sample is the mean per call within one round, followed by the maximum across ranks. Standard deviation and range describe the round means, not request latency percentiles. Positive latency reduction means faster; negative means slower.

The four contrasts compare implementations. They are not independent additive attribution: packing, output strides, and memory copies can interact. Layout-only keeps two gathers; packing-only keeps batch-first gather layout; head-contiguous materializes the head's K/V views.

## Run 9707225: 4 GPUs

Hardware: NVIDIA A100-SXM4-40GB. Python 3.12.13; PyTorch 2.13.0+cu130; CUDA 13.0; NCCL 2.29.7.

Base: `3d43571b94f1683412023c45bd886f9ce30f76bd`. Head: `a2cae127c2933f0e754d38d3615da58660c952f1`.

Each arm: 6 rounds, 10 warmups and 30 measured calls per sample. Validation: **PASS**. Source file hashes and individual samples are in `summary.json` and the raw artifacts.

### B=2, global S=4096, contiguous

Exact Q/K/V equality: True. Largest attention error: 0; changed-input graph replay error: 0.

| Arm | Gather calls (expected) | Preparation peak increment, MiB | Retained increment, MiB |
|---|---:|---:|---:|
| baseline | 2 (2) | 192.0 | 128.0 |
| layout_only | 2 (2) | 144.0 | 128.0 |
| packing_only | 1 (1) | 288.0 | 128.0 |
| head | 1 (1) | 160.0 | 128.0 |
| head_contiguous | 1 (1) | 256.0 | 128.0 |

Memory is eager preparation's allocated increment over entry allocation, with validation outputs freed; it is not total model memory.

**eager_pre_attention**, GPU milliseconds

| Arm | Mean | Sample std | Median | Min–max |
|---|---:|---:|---:|---:|
| baseline | 0.993445 | 0.067577 | 0.952286 | 0.950921–1.107797 |
| layout_only | 0.747173 | 0.035750 | 0.733116 | 0.730863–0.820122 |
| packing_only | 0.953583 | 0.046614 | 0.935185 | 0.932898–1.048713 |
| head | 0.727063 | 0.035017 | 0.711953 | 0.711817–0.798447 |
| head_contiguous | 1.054219 | 0.057749 | 1.020928 | 1.015091–1.150839 |

**eager_pre_plus_attention**, GPU milliseconds

| Arm | Mean | Sample std | Median | Min–max |
|---|---:|---:|---:|---:|
| baseline | 1.648782 | 0.000267 | 1.648845 | 1.648299–1.649084 |
| layout_only | 1.425436 | 0.000627 | 1.425493 | 1.424520–1.426227 |
| packing_only | 1.623370 | 0.001408 | 1.623927 | 1.620719–1.624508 |
| head | 1.410816 | 0.003823 | 1.410167 | 1.407147–1.415885 |
| head_contiguous | 1.712953 | 0.000863 | 1.712947 | 1.711957–1.714176 |

**graph_pre_plus_attention**, GPU milliseconds

| Arm | Mean | Sample std | Median | Min–max |
|---|---:|---:|---:|---:|
| baseline | 1.643452 | 0.001422 | 1.643674 | 1.640926–1.645090 |
| layout_only | 1.425692 | 0.000899 | 1.425545 | 1.424862–1.427354 |
| packing_only | 1.623592 | 0.002008 | 1.623450 | 1.620890–1.626487 |
| head | 1.410293 | 0.001522 | 1.409980 | 1.408751–1.412335 |
| head_contiguous | 1.709164 | 0.000722 | 1.709193 | 1.708271–1.710012 |

| Contrast | Eager preparation | Eager preparation + attention | Graph preparation + attention |
|---|---:|---:|---:|
| baseline -> layout_only | +24.79% | +13.55% | +13.25% |
| baseline -> packing_only | +4.01% | +1.54% | +1.21% |
| layout_only -> head | +2.69% | +1.03% | +1.08% |
| head -> head_contiguous | -45.00% | -21.42% | -21.19% |

### B=2, global S=4096, projection_view

Exact Q/K/V equality: True. Largest attention error: 0; changed-input graph replay error: 0.

| Arm | Gather calls (expected) | Preparation peak increment, MiB | Retained increment, MiB |
|---|---:|---:|---:|
| baseline | 2 (2) | 192.0 | 128.0 |
| layout_only | 2 (2) | 144.0 | 128.0 |
| packing_only | 1 (1) | 288.0 | 128.0 |
| head | 1 (1) | 160.0 | 128.0 |
| head_contiguous | 1 (1) | 256.0 | 128.0 |

Memory is eager preparation's allocated increment over entry allocation, with validation outputs freed; it is not total model memory.

**eager_pre_attention**, GPU milliseconds

| Arm | Mean | Sample std | Median | Min–max |
|---|---:|---:|---:|---:|
| baseline | 1.032841 | 0.002186 | 1.032557 | 1.030281–1.036756 |
| layout_only | 0.752520 | 0.047197 | 0.733389 | 0.732391–0.848850 |
| packing_only | 1.010543 | 0.001532 | 1.011087 | 1.008090–1.012053 |
| head | 0.713174 | 0.000995 | 0.713244 | 0.711898–0.714464 |
| head_contiguous | 1.017894 | 0.002508 | 1.017043 | 1.016057–1.022681 |

**eager_pre_plus_attention**, GPU milliseconds

| Arm | Mean | Sample std | Median | Min–max |
|---|---:|---:|---:|---:|
| baseline | 1.726841 | 0.000800 | 1.726892 | 1.725466–1.727715 |
| layout_only | 1.429298 | 0.002756 | 1.429108 | 1.425627–1.432929 |
| packing_only | 1.706508 | 0.002135 | 1.707341 | 1.702708–1.708223 |
| head | 1.409341 | 0.001639 | 1.408628 | 1.408061–1.412307 |
| head_contiguous | 1.713489 | 0.002030 | 1.713713 | 1.709897–1.715350 |

**graph_pre_plus_attention**, GPU milliseconds

| Arm | Mean | Sample std | Median | Min–max |
|---|---:|---:|---:|---:|
| baseline | 1.724115 | 0.000902 | 1.724185 | 1.723136–1.725389 |
| layout_only | 1.428648 | 0.001425 | 1.429198 | 1.426763–1.430136 |
| packing_only | 1.705468 | 0.001996 | 1.705640 | 1.702496–1.708378 |
| head | 1.410046 | 0.002454 | 1.409661 | 1.407390–1.414462 |
| head_contiguous | 1.708999 | 0.001173 | 1.708744 | 1.707747–1.711065 |

| Contrast | Eager preparation | Eager preparation + attention | Graph preparation + attention |
|---|---:|---:|---:|
| baseline -> layout_only | +27.14% | +17.23% | +17.14% |
| baseline -> packing_only | +2.16% | +1.18% | +1.08% |
| layout_only -> head | +5.23% | +1.40% | +1.30% |
| head -> head_contiguous | -42.73% | -21.58% | -21.20% |

### B=2, global S=16384, contiguous

Exact Q/K/V equality: True. Largest attention error: 0; changed-input graph replay error: 0.

| Arm | Gather calls (expected) | Preparation peak increment, MiB | Retained increment, MiB |
|---|---:|---:|---:|
| baseline | 2 (2) | 768.0 | 512.0 |
| layout_only | 2 (2) | 576.0 | 512.0 |
| packing_only | 1 (1) | 1152.0 | 512.0 |
| head | 1 (1) | 640.0 | 512.0 |
| head_contiguous | 1 (1) | 1024.0 | 512.0 |

Memory is eager preparation's allocated increment over entry allocation, with validation outputs freed; it is not total model memory.

**eager_pre_attention**, GPU milliseconds

| Arm | Mean | Sample std | Median | Min–max |
|---|---:|---:|---:|---:|
| baseline | 3.421499 | 0.002236 | 3.421928 | 3.418495–3.424731 |
| layout_only | 2.552483 | 0.001473 | 2.552558 | 2.550976–2.554958 |
| packing_only | 3.525330 | 0.008237 | 3.522170 | 3.519294–3.541039 |
| head | 2.640019 | 0.002017 | 2.639889 | 2.637111–2.643263 |
| head_contiguous | 3.816556 | 0.002494 | 3.816458 | 3.813709–3.820795 |

**eager_pre_plus_attention**, GPU milliseconds

| Arm | Mean | Sample std | Median | Min–max |
|---|---:|---:|---:|---:|
| baseline | 13.761637 | 0.018234 | 13.755137 | 13.745320–13.788428 |
| layout_only | 12.922415 | 0.011200 | 12.924485 | 12.902287–12.933339 |
| packing_only | 13.818537 | 0.032633 | 13.829071 | 13.761907–13.847003 |
| head | 13.020603 | 0.021879 | 13.018229 | 12.993649–13.054728 |
| head_contiguous | 14.104374 | 0.021264 | 14.111728 | 14.065108–14.123508 |

**graph_pre_plus_attention**, GPU milliseconds

| Arm | Mean | Sample std | Median | Min–max |
|---|---:|---:|---:|---:|
| baseline | 13.756216 | 0.006067 | 13.754314 | 13.750656–13.766015 |
| layout_only | 12.928762 | 0.006232 | 12.928141 | 12.922417–12.936384 |
| packing_only | 13.837613 | 0.009997 | 13.835167 | 13.824526–13.850348 |
| head | 13.026083 | 0.015220 | 13.025293 | 13.008112–13.045281 |
| head_contiguous | 14.121651 | 0.014376 | 14.122706 | 14.098016–14.139878 |

| Contrast | Eager preparation | Eager preparation + attention | Graph preparation + attention |
|---|---:|---:|---:|
| baseline -> layout_only | +25.40% | +6.10% | +6.02% |
| baseline -> packing_only | -3.03% | -0.41% | -0.59% |
| layout_only -> head | -3.43% | -0.76% | -0.75% |
| head -> head_contiguous | -44.57% | -8.32% | -8.41% |

### B=1, global S=4096, contiguous

Exact Q/K/V equality: True. Largest attention error: 0; changed-input graph replay error: 0.

| Arm | Gather calls (expected) | Preparation peak increment, MiB | Retained increment, MiB |
|---|---:|---:|---:|
| baseline | 2 (2) | 64.0 | 64.0 |
| layout_only | 2 (2) | 64.0 | 64.0 |
| packing_only | 1 (1) | 80.0 | 64.0 |
| head | 1 (1) | 80.0 | 64.0 |
| head_contiguous | 1 (1) | 128.0 | 64.0 |

Memory is eager preparation's allocated increment over entry allocation, with validation outputs freed; it is not total model memory.

**eager_pre_attention**, GPU milliseconds

| Arm | Mean | Sample std | Median | Min–max |
|---|---:|---:|---:|---:|
| baseline | 0.359209 | 0.001985 | 0.358458 | 0.357925–0.363188 |
| layout_only | 0.378713 | 0.029439 | 0.361078 | 0.358380–0.416955 |
| packing_only | 0.354904 | 0.003269 | 0.353994 | 0.351543–0.359696 |
| head | 0.378479 | 0.032679 | 0.366145 | 0.350513–0.419582 |
| head_contiguous | 0.500631 | 0.003382 | 0.499343 | 0.498772–0.507479 |

**eager_pre_plus_attention**, GPU milliseconds

| Arm | Mean | Sample std | Median | Min–max |
|---|---:|---:|---:|---:|
| baseline | 0.790250 | 0.000782 | 0.790656 | 0.788800–0.790795 |
| layout_only | 0.790742 | 0.001175 | 0.790796 | 0.789483–0.791922 |
| packing_only | 0.779301 | 0.000804 | 0.779292 | 0.778335–0.780556 |
| head | 0.779071 | 0.000882 | 0.778875 | 0.777909–0.780254 |
| head_contiguous | 0.927034 | 0.001012 | 0.926847 | 0.926126–0.928395 |

**graph_pre_plus_attention**, GPU milliseconds

| Arm | Mean | Sample std | Median | Min–max |
|---|---:|---:|---:|---:|
| baseline | 0.788380 | 0.000801 | 0.788460 | 0.786935–0.789151 |
| layout_only | 0.788894 | 0.000772 | 0.788763 | 0.788176–0.790339 |
| packing_only | 0.778114 | 0.001548 | 0.778204 | 0.776018–0.780169 |
| head | 0.777329 | 0.000944 | 0.777419 | 0.775813–0.778542 |
| head_contiguous | 0.922481 | 0.000636 | 0.922377 | 0.921545–0.923284 |

| Contrast | Eager preparation | Eager preparation + attention | Graph preparation + attention |
|---|---:|---:|---:|
| baseline -> layout_only | -5.43% | -0.06% | -0.07% |
| baseline -> packing_only | +1.20% | +1.39% | +1.30% |
| layout_only -> head | +0.06% | +1.48% | +1.47% |
| head -> head_contiguous | -32.27% | -18.99% | -18.67% |

### Untimed profile: PASS

Counts cover three untimed pre_attention calls on rank 0. Nested events overlap; no durations are summed.

| Arm | Status | NCCL CUDA kernel events | aten::clone | aten::copy_ | aten::contiguous |
|---|---|---:|---:|---:|---:|
| baseline | PASS | 6 | 6 | 6 | 0 |
| layout_only | PASS | 6 | 6 | 6 | 6 |
| packing_only | PASS | 3 | 3 | 3 | 0 |
| head | PASS | 3 | 0 | 0 | 0 |
| head_contiguous | PASS | 3 | 6 | 6 | 6 |

NCCL kernel counts depend on the selected protocol and need not equal collective counts. Operator counts describe observed work; they do not establish how much latency each operation caused.

## Run 9707226: 8 GPUs

Hardware: NVIDIA A100-SXM4-40GB. Python 3.12.13; PyTorch 2.13.0+cu130; CUDA 13.0; NCCL 2.29.7.

Base: `3d43571b94f1683412023c45bd886f9ce30f76bd`. Head: `a2cae127c2933f0e754d38d3615da58660c952f1`.

Each arm: 6 rounds, 10 warmups and 30 measured calls per sample. Validation: **PASS**. Source file hashes and individual samples are in `summary.json` and the raw artifacts.

### B=2, global S=4096, contiguous

Exact Q/K/V equality: True. Largest attention error: 0; changed-input graph replay error: 0.

| Arm | Gather calls (expected) | Preparation peak increment, MiB | Retained increment, MiB |
|---|---:|---:|---:|
| baseline | 2 (2) | 192.0 | 128.0 |
| layout_only | 2 (2) | 136.0 | 128.0 |
| packing_only | 1 (1) | 272.0 | 128.0 |
| head | 1 (1) | 144.0 | 128.0 |
| head_contiguous | 1 (1) | 256.0 | 128.0 |

Memory is eager preparation's allocated increment over entry allocation, with validation outputs freed; it is not total model memory.

**eager_pre_attention**, GPU milliseconds

| Arm | Mean | Sample std | Median | Min–max |
|---|---:|---:|---:|---:|
| baseline | 1.099463 | 0.274486 | 0.974677 | 0.972834–1.656730 |
| layout_only | 0.736785 | 0.050581 | 0.716117 | 0.715332–0.840021 |
| packing_only | 1.004214 | 0.029995 | 0.991983 | 0.991573–1.065438 |
| head | 0.746519 | 0.016615 | 0.736000 | 0.735573–0.768171 |
| head_contiguous | 1.057485 | 0.030760 | 1.038729 | 1.037107–1.107081 |

**eager_pre_plus_attention**, GPU milliseconds

| Arm | Mean | Sample std | Median | Min–max |
|---|---:|---:|---:|---:|
| baseline | 1.416664 | 0.012532 | 1.411942 | 1.410458–1.442202 |
| layout_only | 1.147876 | 0.001337 | 1.147443 | 1.146573–1.149815 |
| packing_only | 1.419474 | 0.000745 | 1.419622 | 1.418445–1.420425 |
| head | 1.175114 | 0.012504 | 1.170091 | 1.169237–1.200606 |
| head_contiguous | 1.471414 | 0.001103 | 1.471556 | 1.470123–1.473126 |

**graph_pre_plus_attention**, GPU milliseconds

| Arm | Mean | Sample std | Median | Min–max |
|---|---:|---:|---:|---:|
| baseline | 1.406003 | 0.001050 | 1.405867 | 1.404382–1.407215 |
| layout_only | 1.146766 | 0.001535 | 1.146214 | 1.145890–1.149884 |
| packing_only | 1.417353 | 0.000614 | 1.417182 | 1.416738–1.418206 |
| head | 1.172184 | 0.000349 | 1.172156 | 1.171797–1.172617 |
| head_contiguous | 1.469616 | 0.000463 | 1.469730 | 1.469030–1.470259 |

| Contrast | Eager preparation | Eager preparation + attention | Graph preparation + attention |
|---|---:|---:|---:|
| baseline -> layout_only | +32.99% | +18.97% | +18.44% |
| baseline -> packing_only | +8.66% | -0.20% | -0.81% |
| layout_only -> head | -1.32% | -2.37% | -2.22% |
| head -> head_contiguous | -41.66% | -25.21% | -25.37% |

### B=2, global S=4096, projection_view

Exact Q/K/V equality: True. Largest attention error: 0; changed-input graph replay error: 0.

| Arm | Gather calls (expected) | Preparation peak increment, MiB | Retained increment, MiB |
|---|---:|---:|---:|
| baseline | 2 (2) | 192.0 | 128.0 |
| layout_only | 2 (2) | 136.0 | 128.0 |
| packing_only | 1 (1) | 272.0 | 128.0 |
| head | 1 (1) | 144.0 | 128.0 |
| head_contiguous | 1 (1) | 256.0 | 128.0 |

Memory is eager preparation's allocated increment over entry allocation, with validation outputs freed; it is not total model memory.

**eager_pre_attention**, GPU milliseconds

| Arm | Mean | Sample std | Median | Min–max |
|---|---:|---:|---:|---:|
| baseline | 1.015694 | 0.001051 | 1.015569 | 1.014613–1.017276 |
| layout_only | 0.717540 | 0.000435 | 0.717687 | 0.716937–0.717926 |
| packing_only | 1.032562 | 0.000568 | 1.032311 | 1.032090–1.033318 |
| head | 0.737058 | 0.002178 | 0.736273 | 0.735812–0.741444 |
| head_contiguous | 1.038137 | 0.000748 | 1.038029 | 1.037346–1.039360 |

**eager_pre_plus_attention**, GPU milliseconds

| Arm | Mean | Sample std | Median | Min–max |
|---|---:|---:|---:|---:|
| baseline | 1.452686 | 0.001266 | 1.452544 | 1.451452–1.454353 |
| layout_only | 1.150982 | 0.003604 | 1.149696 | 1.147870–1.156881 |
| packing_only | 1.466721 | 0.004481 | 1.465020 | 1.464081–1.475721 |
| head | 1.170625 | 0.001035 | 1.170569 | 1.169101–1.172002 |
| head_contiguous | 1.473394 | 0.004122 | 1.472239 | 1.470464–1.481660 |

**graph_pre_plus_attention**, GPU milliseconds

| Arm | Mean | Sample std | Median | Min–max |
|---|---:|---:|---:|---:|
| baseline | 1.448670 | 0.000759 | 1.448397 | 1.448175–1.450155 |
| layout_only | 1.148376 | 0.000782 | 1.148570 | 1.147187–1.149303 |
| packing_only | 1.464428 | 0.000856 | 1.464781 | 1.462750–1.465105 |
| head | 1.168782 | 0.000693 | 1.168896 | 1.167599–1.169544 |
| head_contiguous | 1.467637 | 0.000664 | 1.467477 | 1.466880–1.468723 |

| Contrast | Eager preparation | Eager preparation + attention | Graph preparation + attention |
|---|---:|---:|---:|
| baseline -> layout_only | +29.35% | +20.77% | +20.73% |
| baseline -> packing_only | -1.66% | -0.97% | -1.09% |
| layout_only -> head | -2.72% | -1.71% | -1.78% |
| head -> head_contiguous | -40.85% | -25.86% | -25.57% |

### B=2, global S=16384, contiguous

Exact Q/K/V equality: True. Largest attention error: 0; changed-input graph replay error: 0.

| Arm | Gather calls (expected) | Preparation peak increment, MiB | Retained increment, MiB |
|---|---:|---:|---:|
| baseline | 2 (2) | 768.0 | 512.0 |
| layout_only | 2 (2) | 544.0 | 512.0 |
| packing_only | 1 (1) | 1088.0 | 512.0 |
| head | 1 (1) | 576.0 | 512.0 |
| head_contiguous | 1 (1) | 1024.0 | 512.0 |

Memory is eager preparation's allocated increment over entry allocation, with validation outputs freed; it is not total model memory.

**eager_pre_attention**, GPU milliseconds

| Arm | Mean | Sample std | Median | Min–max |
|---|---:|---:|---:|---:|
| baseline | 3.617490 | 0.001689 | 3.617690 | 3.614515–3.619567 |
| layout_only | 2.602758 | 0.001838 | 2.602138 | 2.601028–2.605704 |
| packing_only | 3.612439 | 0.001400 | 3.611972 | 3.611238–3.615164 |
| head | 2.592586 | 0.001433 | 2.592222 | 2.591095–2.594714 |
| head_contiguous | 3.763718 | 0.002077 | 3.763558 | 3.760913–3.767364 |

**eager_pre_plus_attention**, GPU milliseconds

| Arm | Mean | Sample std | Median | Min–max |
|---|---:|---:|---:|---:|
| baseline | 8.933262 | 0.007043 | 8.934041 | 8.922044–8.942592 |
| layout_only | 7.905963 | 0.004394 | 7.903744 | 7.902856–7.913711 |
| packing_only | 8.926424 | 0.007578 | 8.924996 | 8.917504–8.940032 |
| head | 7.908858 | 0.002242 | 7.908625 | 7.905587–7.912345 |
| head_contiguous | 9.083660 | 0.003045 | 9.083119 | 9.080593–9.088580 |

**graph_pre_plus_attention**, GPU milliseconds

| Arm | Mean | Sample std | Median | Min–max |
|---|---:|---:|---:|---:|
| baseline | 8.922937 | 0.004313 | 8.923494 | 8.916582–8.927642 |
| layout_only | 7.903989 | 0.004887 | 7.902908 | 7.899853–7.912311 |
| packing_only | 8.926015 | 0.003138 | 8.926481 | 8.921634–8.929143 |
| head | 7.910798 | 0.002591 | 7.909751 | 7.908864–7.915486 |
| head_contiguous | 9.078426 | 0.003599 | 9.077197 | 9.074688–9.083358 |

| Contrast | Eager preparation | Eager preparation + attention | Graph preparation + attention |
|---|---:|---:|---:|
| baseline -> layout_only | +28.05% | +11.50% | +11.42% |
| baseline -> packing_only | +0.14% | +0.08% | -0.03% |
| layout_only -> head | +0.39% | -0.04% | -0.09% |
| head -> head_contiguous | -45.17% | -14.85% | -14.76% |

### B=1, global S=4096, contiguous

Exact Q/K/V equality: True. Largest attention error: 0; changed-input graph replay error: 0.

| Arm | Gather calls (expected) | Preparation peak increment, MiB | Retained increment, MiB |
|---|---:|---:|---:|
| baseline | 2 (2) | 64.0 | 64.0 |
| layout_only | 2 (2) | 64.0 | 64.0 |
| packing_only | 1 (1) | 72.0 | 64.0 |
| head | 1 (1) | 72.0 | 64.0 |
| head_contiguous | 1 (1) | 128.0 | 64.0 |

Memory is eager preparation's allocated increment over entry allocation, with validation outputs freed; it is not total model memory.

**eager_pre_attention**, GPU milliseconds

| Arm | Mean | Sample std | Median | Min–max |
|---|---:|---:|---:|---:|
| baseline | 0.436491 | 0.000234 | 0.436548 | 0.436122–0.436702 |
| layout_only | 0.436395 | 0.000195 | 0.436378 | 0.436122–0.436702 |
| packing_only | 0.349980 | 0.000506 | 0.350123 | 0.349013–0.350481 |
| head | 0.351494 | 0.003533 | 0.350106 | 0.349730–0.358673 |
| head_contiguous | 0.498631 | 0.000667 | 0.498398 | 0.498210–0.499951 |

**eager_pre_plus_attention**, GPU milliseconds

| Arm | Mean | Sample std | Median | Min–max |
|---|---:|---:|---:|---:|
| baseline | 0.732621 | 0.015865 | 0.726084 | 0.725026–0.764928 |
| layout_only | 0.731853 | 0.012532 | 0.727313 | 0.724992–0.757350 |
| packing_only | 0.631342 | 0.000607 | 0.631159 | 0.630852–0.632491 |
| head | 0.631285 | 0.001644 | 0.631876 | 0.629009–0.633173 |
| head_contiguous | 0.784435 | 0.001788 | 0.783838 | 0.782950–0.787524 |

**graph_pre_plus_attention**, GPU milliseconds

| Arm | Mean | Sample std | Median | Min–max |
|---|---:|---:|---:|---:|
| baseline | 0.723661 | 0.001875 | 0.722995 | 0.722364–0.727313 |
| layout_only | 0.725743 | 0.000490 | 0.725555 | 0.725265–0.726528 |
| packing_only | 0.631211 | 0.000917 | 0.630989 | 0.630340–0.632934 |
| head | 0.629367 | 0.001025 | 0.629675 | 0.627575–0.630306 |
| head_contiguous | 0.782382 | 0.001434 | 0.781961 | 0.781449–0.785271 |

| Contrast | Eager preparation | Eager preparation + attention | Graph preparation + attention |
|---|---:|---:|---:|
| baseline -> layout_only | +0.02% | +0.10% | -0.29% |
| baseline -> packing_only | +19.82% | +13.82% | +12.78% |
| layout_only -> head | +19.46% | +13.74% | +13.28% |
| head -> head_contiguous | -41.86% | -24.26% | -24.31% |

### Untimed profile: NOT_REQUESTED_OR_NO_ARTIFACT
