import pandas as pd

psnr_hvsm_np = pd.read_csv('tid2013_results/numpy/psnr_hvsm.txt', header=None)
psnr_hvsm_pt = pd.read_csv('tid2013_results/torch/psnr_hvsm.txt', header=None)
psnr_hvsm_cpp = pd.read_csv('tid2013_results/numpy/psnr_hvsm.txt', header=None)
psnr_hvsm_ref = pd.read_csv('D:/Devel/tid2013/metrics_values/PSNRHVSM.txt', header=None)

psnr_hma_np = pd.read_csv('tid2013_results/numpy/psnr_hma.txt', header=None)
psnr_hma_pt = pd.read_csv('tid2013_results/torch/psnr_hma.txt', header=None)
psnr_hma_cpp = pd.read_csv('tid2013_results/numpy/psnr_hma.txt', header=None)
psnr_hma_ref = pd.read_csv('D:/Devel/tid2013/metrics_values/PSNRHMA.txt', header=None)

results = pd.DataFrame({
    'torch': psnr_hvsm_pt.to_numpy().clip(0, 100).flat,
    'numpy': psnr_hvsm_np.to_numpy().clip(0, 100).flat,
    'cpp': psnr_hvsm_cpp.to_numpy().clip(0, 100).flat,
    'ref': psnr_hvsm_ref.clip(0, 100).to_numpy().flat
})
results = results.eval('dif_numpy = abs(numpy-ref)')
results = results.eval('dif_torch = abs(torch-ref)')
results = results.eval('dif_cpp = abs(cpp-ref)')

print(results.query('dif_numpy>1e-3'))
print(results.query('dif_torch>1e-3'))
print(results.query('dif_cpp>1e-3'))
