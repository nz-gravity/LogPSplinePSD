import os
task_id = int(os.environ.get('SLURM_ARRAY_TASK_ID', 0))
print(f"Current task ID: {task_id}")
import numpy as np
import pandas as pd
from log_psplines.datatypes import MultivariateTimeseries
from log_psplines.mcmc import run_mcmc
from log_psplines.example_datasets.varma_data import _calculate_true_varma_psd
import time

n = 256
sigma = np.array([[1., 0.9], [0.9, 1.]])  
varCoef = np.array([[[0.5, 0.], [0., -0.3]], [[0., 0.], [0., -0.5]]])
vmaCoef = np.array([[[1.,0.],[0.,1.]]])

# load data
data_whole = pd.read_csv("var2_256_data.csv")
data_whole = data_whole.values

iteration_start_time = time.time()

y = data_whole[task_id*n:((task_id+1)*n),:]

# estimate var2 PSD by p-spline
t = np.arange(n, dtype=float)
ts = MultivariateTimeseries(t=t, y=y)

idata = run_mcmc(
    data=ts,
    sampler="multivar_blocked_nuts",
    n_knots=25,
    degree=3,
    diffMatrixOrder=2,
    init_from_vi=True,
    n_samples=1000,
    n_warmup=1000,
    rng_key=task_id,
    verbose=True,
    outdir=None
)

freq_old = np.asarray(idata.posterior_psd["freq"].values)
freq = freq_old*2*np.pi
psd_real = idata.posterior_psd["psd_matrix_real"].values
psd_imag = idata.posterior_psd["psd_matrix_imag"].values

psd_imag_mod = psd_imag.copy()
nch = psd_imag_mod.shape[-1]
idx = np.arange(nch)
psd_imag_mod[..., idx, idx] = 0.0

estimated_psd = psd_real.astype(np.complex128) + 1j * psd_imag_mod.astype(np.complex128)
estimated_psd = estimated_psd/(4*np.pi)
est_lower, est_med, est_upper = estimated_psd

# find true var2 PSD
fs = 1.0
true_psd_full = _calculate_true_varma_psd(
    freqs=freq_old,
    dim=2,
    var_coeffs=varCoef,
    vma_coeffs=vmaCoef,
    sigma=sigma,
    fs=fs,
)
true_psd_full[..., idx, idx] = np.real(true_psd_full[..., idx, idx])
true_psd = true_psd_full/(2*np.pi)


#find L1 error--------------------------------------------------------------------------
L1_error = np.empty(n//2)
for i in range(n//2):
    L1_error[i] = np.sqrt(np.sum(np.diag((est_med[i,:,:]-true_psd[i,:,:]) @
                                (est_med[i,:,:]-true_psd[i,:,:]))))
L1_p_spline = np.mean(L1_error)

#find L2 error---------------------------------------------------------------------------
L2_error = np.empty(n//2)
for i in range(n//2):
    L2_error[i] = np.sum(np.diag((est_med[i,:,:]-true_psd[i,:,:]) @
                                (est_med[i,:,:]-true_psd[i,:,:])))
L2_p_spline = np.sqrt(np.mean(L2_error))

##find length of pointwise CI-------------------------------------------------------------
len_CI_f11 = (np.median(np.real(est_upper[:,0,0])) - np.median(np.real(est_lower[:,0,0])))
len_CI_re_f12 = (np.median(np.real(est_upper[:,0,1])) - np.median(np.real(est_lower[:,0,1])))
len_CI_im_f12 = (np.median(np.imag(est_upper[:,1,0])) - np.median(np.imag(est_lower[:,1,0])))
len_CI_f22 = (np.median(np.real(est_upper[:,1,1])) - np.median(np.real(est_lower[:,1,1])))

##find coverage for pointwise CI----------------------------------------------------------
def complex_to_real(matrix):
    n = matrix.shape[0]
    real_matrix = np.zeros_like(matrix, dtype=float)
    real_matrix[np.triu_indices(n)] = np.real(matrix[np.triu_indices(n)])
    real_matrix[np.tril_indices(n, -1)] = np.imag(matrix[np.tril_indices(n, -1)])
    
    return real_matrix

est_lower_real = np.zeros_like(est_lower, dtype=float)
est_upper_real = np.zeros_like(est_upper, dtype=float)
true_psd_real = np.zeros_like(true_psd, dtype=float)
for j in range(len(freq)):
    est_lower_real[j] = complex_to_real(est_lower[j])
    est_upper_real[j] = complex_to_real(est_upper[j])
    true_psd_real[j] = complex_to_real(true_psd[j])

coverage_point_CI = np.mean((est_lower_real <= true_psd_real) & (true_psd_real <= est_upper_real))

iteration_end_time = time.time()
total_iteration_time = iteration_end_time - iteration_start_time

## collect all results--------------------------------------------------------------------
results = {
    'task_id': task_id,
    'L1_p_spline': L1_p_spline,
    'L2_p_spline': L2_p_spline,
    'len_CI_f11': len_CI_f11,
    'len_CI_re_f12': len_CI_re_f12,
    'len_CI_im_f12': len_CI_im_f12,
    'len_CI_f22': len_CI_f22,
    'coverage_pointwise': coverage_point_CI,
    'total_iteration_time': total_iteration_time
}

result_df = pd.DataFrame([results])
csv_file = f'var2_256_{task_id}.csv'
result_df.to_csv(csv_file, index=False)








