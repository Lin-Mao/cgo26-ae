# AE of PASTA (CGO'26)



## Installation

* Dependencies
  * PyTorch >= 2.0
  * CUDA >= 12.0

* Download the codebase

  ```shell
  git clone --recursive https://github.com/AccelProf/AccelProf.git
  cd AccelProf && git checkout cgo26
  git submodule update --init --recursive
  
  # compile code
  make ENABLE_CS=1 ENABLE_NVBIT=1 ENABLE_TORCH=1
  
  # Set env, under AccelProf directory
  export ACCEL_PROF_HOME=$(pwd)
  export PATH=${ACCEL_PROF_HOME}/bin:${PATH}
  
  # download AE package
  bash bin/setup_ae
  ```

* Setup Artifact

  ```shell
  cd cgo26-ae
  # download benchmark
  bash ./bin/setup_artifact.sh
  ```

## Experiments


### Figure 7

```shell
bash ./bin/run_figure_7.sh
```

Result is in  `results/figure_7/figure7.pdf`.

What's expected: We're expected to see there're some kernels (top-20 are shown) are repeated innovated.

### Table V

```shell
bash ./bin/run_table_v.sh
```

Result is in ``results/table_v/table_v.log`.

We expected that memory footprint is larger than working set sizes.

### Figure 9 & 10


* Figure 9

Collect the overhead

```shell
bash ./bin/run_figure_9.sh
# log is generated in raw_data/figure_9
```

Plot Figure 9

```shell
bash ./bin/plot_figure_9.sh raw_data/figure_9 
```

* Figure 10

Collect the overhead breakdown

```shell
bash ./bin/run_figure_10.sh
```

Plot Figure 10

```shell
bash ./bin/plot_figure_10.sh raw_data/figure_10
```



### Figure 11 & 12

Figure 11 and 12 shows the case study that demonstrate the capability of PASTA can capture tensor-level information. We use this information to guide UVM prefetch. To avoid UVM drivers internal prefetch behaviors, we disable it by modifying the NVIDIA open gpu kernel modules (https://github.com/NVIDIA/open-gpu-kernel-modules). We suggest reviewers evalute these two on the provided machine, which has been already set up.


* Run experiments

```shell
# Run profile
bash bin/run_uvm_profiling.sh

# you can specify number of runs, we run 5 times and get the average
bash bin/run_figure_11.sh 5
bash bin/run_figure_12.sh 5
```

### Figure 13

Run

```shell
bash bin/run_figure_13.sh
```



### Figure 14

Run

```shell
bash bin/run_figure_14.sh
```



### Figure 15

```shell
accelprof -v -t event_trace_mgpu ./examples/gpt3/run_dist_training.sh
```

