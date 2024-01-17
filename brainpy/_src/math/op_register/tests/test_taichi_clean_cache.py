import brainpy.math as bm

print(bm.check_kernels_count())

bm.clean_caches()

print(bm.check_kernels_count())