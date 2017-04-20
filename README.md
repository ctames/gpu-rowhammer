# gpu-rowhammer
This repository is for sharing code and information related to researching the "rowhammer" problem with respect to GPUs.

Repository breakdown:
* cudahammer: contains adaptations of google's rowhammer test to GPUs 
* hammering: beginning of a user-facing CUDA rowhammer tests suite (incomplete). goal was to have to ability to set up tests with a variety of data patterns and access patterns
* indirect-ub: files related to Sreepathi's benchmark
* indirect_hammer: this is where the bulk of the work is. contains a variety of different possible approachs to "indirect" hammering via evading cache through forced misses.
* quicktest_ldcv: some small tests revisiting how to possibly use ld.cv, which didn't really work
* raghavendras_files: a few early files from raghavendra

Indirect_hammer breakdown:
* indirect_hammer.cu: a test file that I would use to try very specific setups and techniques without editing one of the main test approaches in the "tests" directory
* tests: contains subdirectories for the various possible hammering approaches. assoc contains tests related to trying to exploit associativity (assoc_variants.cu itself has 5 or 6 separate test kernels). loadXMB contains the microbenchmark that is evaluated in the paper (load3MB is less customizable version of it). cachehog was the beginnings of the approach in which loadXMB was combined with assoc. readwrite contains a quick test to see if cache could be evaded through repeated read-writes to the same address
