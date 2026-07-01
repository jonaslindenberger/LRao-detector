import torch
import numpy as np
import matplotlib.pyplot as plt
import scipy
import numpy as np
from sklearn.model_selection import RepeatedKFold
from tqdm.auto import tqdm
import time
import random
device = "cuda" if torch.cuda.is_available() else "cpu"

class Trafo(torch.nn.Module):
    """
    Transformation (CNN) that maximizes LFI at its output.
    """
    def __init__(self,config):
        super().__init__()
        if "dim_inp_train" in config:
            self.dim_inp = config["dim_inp_train"]
        else:
            self.dim_inp = config["dim_inp"]
        self.da = config["da"]
        bias = False
        layers = []
        layers.append(torch.nn.Conv1d(1, config["n_hidden_channels"], config["filt_size"], bias=bias, padding="same"))
        layers.append(torch.nn.Tanh())
        for _ in range(config["n_conv_layers"]-2):
            layers.append(torch.nn.Conv1d(config["n_hidden_channels"], config["n_hidden_channels"], config["filt_size"], bias=bias, padding="same"))
            layers.append(torch.nn.Tanh())
        layers.append(torch.nn.Conv1d(config["n_hidden_channels"], 1, config["filt_size"], bias=bias, padding="same"))
        layers.append(torch.nn.Flatten())
        self.filt = torch.nn.Sequential(*layers)

        self.filt.apply(init_weights)
        self.psd_method_params = config["psd_method_params"]
    

    def forward(self,x):
        """
        Apply transformation.
        """
        return self.filt(x[:,None,:])

    
    def lfi_diag_dft(self,x,puffer_sides,H=None):
        """
        Estimate the LFI contained in Trafo(x) with respect to additive signals specified by the columns in H using the DFT method.
        """
        if H==None:
            H = torch.eye(self.dim_inp,dtype=x.dtype,device=x.device)
        Pyy = self.est_psd(self.forward(x))
        lfi_diag = torch.zeros(H.shape[1],device=x.device)
        # dMu_ = []
        for i in range(H.shape[1]):
            ds = self.da*H[:,i].reshape(1,-1)
            if puffer_sides>0:
                ds[0,:puffer_sides] = 0
                ds[0,-puffer_sides:] = 0
            x1 = x.clone()
            x1 -= ds
            x2 = x.clone()
            x2 += ds
            dmu = torch.mean((self.forward(x2)-self.forward(x1))/(2*self.da),dim=0)
            dMu = torch.fft.fft(dmu,norm="ortho")
            # dMu_.append(dMu.detach())
            # Pyy_ = (self.est_psd(self.forward(x2))+self.est_psd(self.forward(x1)))/2
            lfi_diag[i] = torch.sum(torch.abs(dMu)**2/Pyy)
        return lfi_diag#, dMu_, Pyy.detach()

    def lfi_diag_dft_train_and_detect(self,x,half_width,H=None):
        """
        Estimate the LFI contained in Trafo(x) with respect to additive signals specified by the columns in H using the DFT method.
        """
        y = self.forward(x)
        Pyy = torch.mean(torch.abs(torch.fft.fft(y.squeeze().unfold(dimension=-1, size=64, step=16), dim=-1,n=x.shape[-1]))**2,dim=0)/64
        dmu0 = self.estimate_averaged_jacobian_H_product(y,torch.eye(y.shape[1],device=x.device),half_width=half_width,epsilon=self.da)#H
        # dft_mat = torch.fft.fft(torch.eye(y.shape[1],device=x.device),norm="ortho")
        # inv_cov = torch.real(torch.conj(dft_mat.T)@(dft_mat/Pyy.reshape(-1,1)))
        dMu0 = torch.fft.fft(dmu0,dim=0,norm="ortho")
        j0 = torch.real(torch.conj(dMu0).T@(dMu0/Pyy.reshape(-1,1)))
        return torch.mean(torch.diag(j0))/H.shape[0]
    
    def est_psd(self,x):
        """
        Estimate PSD from data.
        """
        match self.psd_method_params["name"]:
            case "periodogram":
                return est_psd_averaged_periodogram(x)
            case "yule":
                return est_psd_ar_yule_walker(x,self.psd_method_params["order"])

    def set_statistics_dft(self,x,H=None):
        """
        Compute and set statistics (mean, gradient mean, inv cov mat, LFI) of Trafo(x).
        """
        if H==None:
            H = torch.eye(self.dim_inp,dtype=x.dtype,device=x.device)
        with torch.no_grad():
            y = self.forward(x)
            self.Pyy = self.est_psd(y)
            self.mu0 = 0#torch.mean(y,dim=0,keepdim=True)
            self.dmu0 = torch.zeros(y.shape[1],H.shape[1],device=x.device)
            for i in range(H.shape[1]):
                ds = self.da*H[:,i].reshape(1,-1)
                x1 = x.clone()
                x1 -= ds
                x2 = x.clone()
                x2 += ds
                self.dmu0[:,i] = torch.mean((self.forward(x2)-self.forward(x1))/(2*self.da),dim=0)
        dft_mat = torch.fft.fft(torch.eye(y.shape[1],device=x.device),norm="ortho")
        self.inv_cov = torch.real(torch.conj(dft_mat.T)@(dft_mat/self.Pyy.reshape(-1,1)))
        self.dMu0 = torch.fft.fft(self.dmu0,dim=0,norm="ortho")
        self.j0 = torch.real(torch.conj(self.dMu0).T@(self.dMu0/self.Pyy.reshape(-1,1)))
        self.j0_inv = torch.inverse(self.j0)

    def g(self,x):
        """
        Compute pressimistic score function.
        """
        return self.dmu0.T@self.inv_cov@(self.forward(x)-self.mu0).T

    def estimate(self,x,H=None):
        """
        Compute LBLUE estimate.
        """
        if H==None:
            j0_inv = self.j0_inv
            gx = self.g(x).T
        else:
            j0_inv = torch.inverse(H.T@self.j0@H)
            gx = self.g(x).T@H
        with torch.no_grad():
            return j0_inv@gx.T
    
    def detect(self,x,H=None):
        """
        Compute LRao test statistic.
        """
        with torch.no_grad():
            if H==None:
                j0_inv = self.j0_inv
                gx = self.g(x).T
            else:
                j0_inv = torch.inverse(H.T@self.j0@H)
                gx = self.g(x).T@H
            return torch.concat([gx[i].reshape(1,-1)@j0_inv@gx[i].reshape(-1,1) for i in range(len(x))])

    def calc_test_statistics(self,x,H,theta,H0=True):
        """
        Returns test statistics (t0,t1) computed by LRao.
        """
        theta = (torch.tensor(theta.clone()).reshape(-1,1)).to(x.device)
        s = (H@theta).T
        t1 = self.detect(x+s)
        if H0:
            t0 = self.detect(x)
            return t0,t1
        else:
            return t1
    

def est_psd_averaged_periodogram(x):
    """
    Nonparametric PSD estimate by averaging the periodogram.
    """
    return torch.mean(torch.abs(torch.fft.fft(x,dim=1,norm="ortho"))**2,dim=0)

def est_psd_ar_yule_walker(x,p):
    """
    Autoregressive Parametric PSD estimate using Yule-Walker method.
    """
    a, sigma2 = yule_walker(x, p)
    w = 2*np.pi*torch.arange(x.shape[1],device=x.device).reshape(1,-1)/x.shape[1]
    psda = sigma2/(torch.abs(1-torch.sum(a.reshape(-1,1)*torch.exp(-1j*torch.arange(1,len(a)+1,device=x.device).reshape(-1,1)*w),dim=0))**2)
    return torch.tensor(psda.clone(),dtype=torch.complex64)

def yule_walker(x,order):
    """
    Solve Yule-Walker equations for autoregressive coefficients and noise variance.
    """
    r = torch.zeros((len(x),order+1),dtype=torch.float64,device=x.device)
    r[:,0] = torch.sum(x**2,dim=1)
    for k in range(1, order+1):
        r[:,k] = torch.sum(x[:,0:-k]*x[:,k:],dim=1)
    r = torch.mean(r,dim=0)/x.shape[1]
    R = pytorch_toeplitz(r[:-1].reshape(1,-1))
    a = torch.linalg.solve(R, r[1:])
    sigma2 = r[0] - (r[1:]*a).sum()
    return a, sigma2
    
def pytorch_toeplitz(V):
    """
    Construct Toeplitz matrix from given vector.
    """
    d = V.shape[1]
    A = V.unsqueeze(1).unsqueeze(2)
    A_nofirst_flipped = torch.flip(A[:, :, :, 1:], dims=[3]) 
    A_concat = torch.concatenate([A_nofirst_flipped, A], dim=3) 
    unfold = torch.nn.Unfold(kernel_size=(1, d))
    T = unfold(A_concat)
    T = torch.flip(T, dims=[2])
    return T.squeeze()

def reshape_data(data,dim_inp):
    """
    Reshape data sequence to [n_sequences,dim_inp].
    """
    return data[:(len(data)//dim_inp)*dim_inp].reshape(-1,dim_inp)
    
# def train_data_sim(model,H,H_test,config,noise_gen):
#     """
#     Train model until stopping condition is reached.
#     """
#     match(config["optim"]):
#         case "SGD":
#             optimizer = torch.optim.SGD(model.parameters(), lr=config["lr"],weight_decay=config["weight_decay"])
#         case "AdamW":
#             optimizer = torch.optim.AdamW(model.parameters(), lr=config["lr"],weight_decay=config["weight_decay"])#
#         case "Adam":
#             optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"],weight_decay=config["weight_decay"])
#     model.train()
#     lfi_val = []
#     lfi_train = []
#     data = noise_gen.gen_noise_sequences(config["dim_inp_train"],config["batch_size"])
#     data = data.to(device)
#     break_all = False
#     for i in tqdm(range(config["max_iterations"])):
#         for _ in range(config["iter_batch"]):
#             lfi_diag = model.lfi_diag_dft(data,config["puffer_sides"],H)
#             if i%1==0:
#                 with torch.no_grad():
#                     data_ = noise_gen.gen_noise_sequences(config["dim_inp_val"],config["batch_size_val"])
#                     data_ = data_.to(device)
#                     lfi_val.append(model.lfi_diag_dft(data_,0,H_test).cpu().detach())
#                     print(f"lfi {lfi_val[-1]}")
#             lfi_diag_mean = torch.mean(lfi_diag)
#             lfi_train.append(lfi_diag_mean.cpu().detach())
#             if i>3:
#                 if np.diff([lfi_train[j] for j in range(len(lfi_train))],2)[-1]>0:
#                     print("early stopping")
#                     break_all = True
#                     break
#             loss = -lfi_diag_mean/config["iter_batch"]
#             loss.backward()
#         if break_all:
#             break
#         optimizer.step()
#         optimizer.zero_grad()
#     # data_stats = noise_gen.gen_noise_sequences(config["dim_inp_val"],config["batch_size"])
#     # data_stats = data_stats.to(device)
#     model.eval()
#     model.set_statistics_dft(data.detach(),H_test)
#     return lfi_train, lfi_val, model
def train_data_sim(model,H,H_test,config,noise_gen):
    """
    Train model until stopping condition is reached.
    """
    match(config["optim"]):
        case "SGD":
            optimizer = torch.optim.SGD(model.parameters(), lr=config["lr"],weight_decay=config["weight_decay"])
        case "AdamW":
            optimizer = torch.optim.AdamW(model.parameters(), lr=config["lr"],weight_decay=config["weight_decay"])#
        case "Adam":
            optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"],weight_decay=config["weight_decay"])
    model.train()
    lfi = []
    for i in tqdm(range(config["max_iterations"])):
        for _ in range(config["iter_batch"]):
            data = noise_gen.gen_noise_sequences(config["dim_inp_train"],config["batch_size"])
            data = data.to(device)
            lfi_diag = model.lfi_diag_dft(data,config["puffer_sides"],H)
            if i%10==0:
                with torch.no_grad():
                    data = noise_gen.gen_noise_sequences(config["dim_inp_val"],config["batch_size"])
                    data = data.to(device)
                    lfi.append(model.lfi_diag_dft(data,0,H_test).cpu().detach())
                    print(f"lfi {lfi[-1]}")
            lfi_diag_mean = torch.mean(lfi_diag)
            loss = -lfi_diag_mean/config["iter_batch"]
            loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    data_stats = noise_gen.gen_noise_sequences(config["dim_inp_val"],config["batch_size"])
    data_stats = data_stats.to(device)
    model.eval()
    model.set_statistics_dft(data_stats.detach(),H_test)
    return lfi, model

def train_nested_cv_val(model,data_train,data_val,H,config):
    """
    Train model until stopping condition is reached.
    """
    match(config["optim"]):
        case "SGD":
            optimizer = torch.optim.SGD(model.parameters(), lr=config["lr"],weight_decay=config["weight_decay"])
        case "AdamW":
            optimizer = torch.optim.AdamW(model.parameters(), lr=config["lr"],weight_decay=config["weight_decay"])
        case "Adam":
            optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"],weight_decay=config["weight_decay"])
    model.train()
    lfi_val = []
    patience = 3
    early_stopper = EarlyStopper(patience=patience, min_delta=0)
    for i in range(config["max_iterations"]):
        lfi_diag_mean = torch.mean(model.lfi_diag_dft(data_train,config["puffer_sides"],H))
        loss = -lfi_diag_mean
        loss.backward()
        with torch.no_grad():
            lfi_val.append(torch.mean(model.lfi_diag_dft(data_val,0,H)).cpu())
        optimizer.step()
        optimizer.zero_grad()
        if early_stopper.early_stop(-lfi_val[-1]):
            break
    return max(lfi_val), np.argmax(lfi_val)

def train_nested_cv_test(model,data_train,data_test,H,amplitudes,config,epochs):
    """
    Train and test model for given hyperparameters.
    """
    match(config["optim"]):
        case "SGD":
            optimizer = torch.optim.SGD(model.parameters(), lr=config["lr"],weight_decay=config["weight_decay"])
        case "AdamW":
            optimizer = torch.optim.AdamW(model.parameters(), lr=config["lr"],weight_decay=config["weight_decay"])
        case "Adam":
            optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"],weight_decay=config["weight_decay"])
    model.train()
    lfi = []
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    start_time = time.perf_counter()
    for i in range(epochs):
        lfi_diag_mean = torch.mean(model.lfi_diag_dft(data_train,config["puffer_sides"],H))
        lfi.append(lfi_diag_mean.cpu().detach())
        loss = -lfi_diag_mean
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    model.eval()
    model.set_statistics_dft(data_train.detach(),H)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    end_time = time.perf_counter()
    training_time = end_time - start_time
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    start_time = time.perf_counter()
    t0 = model.detect(data_test).cpu().squeeze()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    end_time = time.perf_counter()
    inference_time = end_time - start_time
    t1 = [None]*len(amplitudes)
    lambda_nc = torch.zeros(len(amplitudes),dtype=data_train.dtype)
    for i in range(len(amplitudes)):
        phases = 2*torch.pi*torch.rand((H.shape[1]//2,1),dtype=data_train.dtype,device=H.device)
        theta = torch.zeros((H.shape[1],1),device=H.device)
        theta[:len(theta)//2] = amplitudes[i]*torch.cos(phases)
        theta[len(theta)//2:] = -amplitudes[i]*torch.sin(phases)
        t1[i] = model.calc_test_statistics(data_test.detach(),H,theta,H0=False).cpu().squeeze()
        lambda_nc[i] = (theta.T@model.j0@theta).cpu().squeeze()
    return t0,torch.stack(t1,dim=0),lambda_nc, lfi, training_time, inference_time
    
def nested_cv(data,H,amplitudes,config_list,n_splits_o,n_repeats_o,n_splits_i,n_repeats_i,n_repeats_phase_net,train_fraction=None):
    """
    Nested cross validation procedure.
    """
    if train_fraction==None:
        train_fraction = 1.0
    t0 = []
    t1 = []
    lambda_nc = []
    lfi = []
    config_best = []
    epochs_best = []
    training_hyper_times = []
    training_times = []
    inference_times = []
    # rkf = RepeatedKFold(n_splits=n_splits_o, n_repeats=n_repeats_o)
    # for train_index, test_index in (pbar:=tqdm(rkf.split(data),total=n_repeats_o*n_splits_o,leave=False)):
    for train_index, test_index in (pbar:=tqdm(customized_repeated_kfold(data, n_splits=n_splits_o, n_repeats=n_repeats_o, train_fraction=train_fraction),total=n_repeats_o*n_splits_o,leave=False)):
        pbar.set_description("outer CV")
        lfi_val = []
        epochs_max_mean = []
        for config in (pbar_config:=tqdm(config_list,leave=False)):
            pbar_config.set_description("hyper params")
            lfi_val_ = []
            epochs_max = []
            rkf = RepeatedKFold(n_splits=n_splits_i, n_repeats=n_repeats_i)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            start_time = time.perf_counter()
            for train_index_i, val_index in (pbar:=tqdm(rkf.split(data[train_index]),total=n_repeats_i*n_splits_i,leave=False)):
                # print(f"inner train: {train_index_i}\ninner val: {val_index}")
                pbar.set_description("inner CV")
                model = Trafo(config)
                model = model.to(device=data.device)
                lfi_val_max, epoch_max = train_nested_cv_val(model,data[train_index_i],data[val_index],H,config)
                lfi_val_.append(lfi_val_max)
                epochs_max.append(epoch_max)
            lfi_val.append(np.mean(lfi_val_))
            epochs_max_mean.append(round(np.mean(epochs_max)))
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            end_time = time.perf_counter()
            training_hyper_times.append(end_time - start_time)
        config_best_ = config_list[np.argmax(lfi_val)]
        config_best.append(config_best_)
        epochs_best_ = epochs_max_mean[np.argmax(lfi_val)]
        epochs_best.append(epochs_best_)
        for i in range(n_repeats_phase_net):
            model = Trafo(config_best_)
            model = model.to(device=data.device)
            t0_,t1_,lambda_nc_, lfi_, training_time, inference_time = train_nested_cv_test(model,data[train_index],data[test_index],H,amplitudes,config_best_,epochs_best_)
            training_times.append(training_time)
            inference_times.append(inference_time)
            t0.append(t0_)
            t1.append(t1_)
            lambda_nc.append(lambda_nc_)
            lfi.append(lfi_)
    return torch.concat(t0), torch.concat(t1,dim=1), torch.mean(torch.stack(lambda_nc),dim=0), lfi, config_best, training_times, training_hyper_times, inference_times

def customized_repeated_kfold(data, n_splits=5, n_repeats=3, train_fraction=0.5, random_state=None):
    """
    Behaves like RepeatedKFold, but splits the training fold into two parts:
    1. A downsampled training set (train_fraction)
    2. An extra testing set (the remainder of the training fold)
    The original validation fold remains completely intact.
    """
    rkf = RepeatedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=random_state)
    
    for train_idx, test_idx in rkf.split(data):
        # Calculate how many samples to keep for training
        keep_size = int(len(train_idx) * train_fraction)
        # print(train_idx)
        random.shuffle(train_idx)
        # print(train_idx)
        # Split the training fold into two distinct parts
        custom_train_idx = train_idx[:keep_size]
        # test_idx = np.concat((test_idx,train_idx[keep_size:]))  # <--- The leftover training data
        # print(f"outer train: {custom_train_idx}\nouter test: {test_idx}")
        
        # Yield all three sets of indices
        yield custom_train_idx, test_idx

def reference_detection(data,H,amplitudes,n_splits_o,n_repeats_o,n_repeats_phase,train_fraction=None):
    """
    Evaluate reference detection methods.
    """
    if train_fraction==None:
        train_fraction = 1.0
    def test_statistics(reference_fun):
        t0 = []
        t1 = []
        # rkf = RepeatedKFold(n_splits=n_splits_o, n_repeats=n_repeats_o)
        # for train_index, test_index in rkf.split(data):
        for train_index, test_index in (pbar:=tqdm(customized_repeated_kfold(data, n_splits=n_splits_o, n_repeats=n_repeats_o, train_fraction=train_fraction),total=n_repeats_o*n_splits_o,leave=False)):
            t0.append(reference_fun(data[train_index],data[test_index],H).cpu().squeeze())
            t1_temp = [None]*len(amplitudes)
            for i in range(len(amplitudes)):
                data_test = []
                for j in range(n_repeats_phase):
                    phases = 2*torch.pi*torch.rand((H.shape[1]//2,1),dtype=torch.float32,device=H.device)
                    theta = torch.zeros((H.shape[1],1),device=H.device)
                    theta[:len(theta)//2] = amplitudes[i]*torch.cos(phases)
                    theta[len(theta)//2:] = -amplitudes[i]*torch.sin(phases)
                    data_test.append(data[test_index]+(H@theta).T)
                t1_temp[i] = reference_fun(data[train_index],torch.concat(data_test,dim=0),H).cpu().squeeze()
            t1.append(torch.stack(t1_temp,dim=0))
        t0 = torch.concat(t0)
        t1 = torch.concat(t1,dim=1)
        return t0, t1

    t0_RaoCGN, t1_RaoCGN = test_statistics(reference_RaoCGN)
    t0_RaoCGN_clipped, t1_RaoCGN_clipped = test_statistics(reference_RaoCGN_clipped)
    t0_RaoCLaplace, t1_RaoCLaplace = test_statistics(reference_RaoCLaplace)
    t0_Rao_adaptive_gn, t1_Rao_adaptive_gn = test_statistics(reference_Rao_adaptive_gn)

    return t0_RaoCGN,t1_RaoCGN,t0_RaoCGN_clipped,t1_RaoCGN_clipped,t0_RaoCLaplace,t1_RaoCLaplace,t0_Rao_adaptive_gn, t1_Rao_adaptive_gn


def reference_RaoCGN(data_train,data_test,H):
    """
    Prewhitening + Rao detector for IID Gaussian noise. Equivalent to GLRT under Gaussian noise assumption.
    """
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    start_time = time.perf_counter()
    dft_mat = torch.fft.fft(torch.eye(data_train.shape[1],device=data_train.device),norm="ortho")
    Pxx = torch.mean(torch.abs(torch.fft.fft(data_train,dim=1,norm="ortho"))**2,dim=0)
    inv_cov = torch.real(torch.conj(dft_mat.T)@(dft_mat/Pxx.reshape(-1,1)))
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    end_time = time.perf_counter()
    training_time = end_time - start_time
    mu = 0#torch.mean(data_train,dim=0)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    start_time = time.perf_counter()
    gx = H.T@inv_cov@(data_test).T#-mu
    j0_inv = torch.inverse(H.T@inv_cov@H)
    res = torch.concat([gx[:,i].reshape(1,-1)@j0_inv@gx[:,i].reshape(-1,1) for i in range(len(data_test))])
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    end_time = time.perf_counter()
    inference_time = end_time - start_time
    return res#, training_time, inference_time

def reference_RaoCGN_clipped(data_train,data_test,H):
    """
    Prewhitening + heuristic Rao detector with limiter function non-linearity.
    """
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    start_time = time.perf_counter()
    clip_lim = 3
    dft_mat = torch.fft.fft(torch.eye(data_train.shape[1],device=data_train.device),norm="ortho")
    Pxx = torch.median(torch.abs(torch.fft.fft(data_train,dim=1,norm="ortho"))**2,dim=0)[0]
    inv_cov = torch.real(torch.conj(dft_mat.T)@(dft_mat/Pxx.reshape(-1,1)))
    inv_cov_sqrt = torch.real(torch.conj(dft_mat.T)@(dft_mat/torch.sqrt(Pxx.reshape(-1,1))))
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    end_time = time.perf_counter()
    training_time = end_time - start_time
    # inv_cov_sqrt = torch.ones_like(inv_cov_sqrt)
    # inv_cov = torch.ones_like(inv_cov)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    start_time = time.perf_counter()
    gx = H.T@inv_cov_sqrt@clip_data(inv_cov_sqrt@(data_test).T,clip_lim,-clip_lim)
    j0_inv = torch.inverse(H.T@inv_cov@H)
    res = torch.concat([gx[:,i].reshape(1,-1)@j0_inv@gx[:,i].reshape(-1,1) for i in range(len(data_test))])
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    end_time = time.perf_counter()
    inference_time = end_time - start_time
    return res#, training_time, inference_time

def reference_RaoCLaplace(data_train,data_test,H):
    """
    Prewhitening + Rao detector for IID Laplace noise.
    """
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    start_time = time.perf_counter()
    dft_mat = torch.fft.fft(torch.eye(data_train.shape[1],device=data_train.device),norm="ortho")
    Pxx = torch.mean(torch.abs(torch.fft.fft(data_train,dim=1,norm="ortho"))**2,dim=0)
    inv_cov_sqrt = torch.real(torch.conj(dft_mat.T)@(dft_mat/torch.sqrt(Pxx).reshape(-1,1)))
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    end_time = time.perf_counter()
    training_time = end_time - start_time
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    start_time = time.perf_counter()
    data_test_ = (inv_cov_sqrt@data_test.T).T
    H_ = inv_cov_sqrt@H
    gx = H_.T@(torch.sign(data_test_)).T
    j0_inv = torch.inverse(H_.T@H_)
    res = torch.concat([gx[:,i].reshape(1,-1)@j0_inv@gx[:,i].reshape(-1,1) for i in range(len(data_test_))])
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    end_time = time.perf_counter()
    inference_time = end_time - start_time
    return res#, training_time, inference_time

def reference_Rao_adaptive_gn(data_train,data_test,H):
    """
    Prewhitening + Rao detector for IID Laplace noise.
    """
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    start_time = time.perf_counter()
    dft_mat = torch.fft.fft(torch.eye(data_train.shape[1],device=data_train.device),norm="ortho")
    Pxx = torch.mean(torch.abs(torch.fft.fft(data_train,dim=1,norm="ortho"))**2,dim=0)
    inv_cov_sqrt = torch.real(torch.conj(dft_mat.T)@(dft_mat/torch.sqrt(Pxx).reshape(-1,1)))
    data_test_ = (inv_cov_sqrt@data_test.T).T
    data_train_ = inv_cov_sqrt@data_train.T
    gx = fit_and_apply_gennorm_score(data_train_, data_test_).T
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    end_time = time.perf_counter()
    training_time = end_time - start_time
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    start_time = time.perf_counter()
    H_ = inv_cov_sqrt@H
    gx = H_.T@gx
    j0_inv = torch.inverse(H_.T@H_)
    res = torch.concat([gx[:,i].reshape(1,-1)@j0_inv@gx[:,i].reshape(-1,1) for i in range(len(data_test_))])
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    end_time = time.perf_counter()
    inference_time = end_time - start_time
    return res#, training_time, inference_time

def fit_and_apply_gennorm_score(data_train_h0, data_test):
    """
    Fits a Generalized Gaussian Distribution (GGD) to the training noise
    and applies the optimal score function to the test data.
    
    Parameters:
        data_train_h0 (numpy array or torch tensor): The H0 training noise.
        data_test (numpy array or torch tensor): The test data (H0 or H1) to transform.
        
    Returns:
        transformed_test_data (torch tensor): The data passed through the LOD.
    """
    # 1. Flatten the training data to fit the global noise distribution
    if isinstance(data_train_h0, torch.Tensor):
        train_flat = data_train_h0.cpu().numpy().flatten()
    else:
        train_flat = data_train_h0.flatten()
        
    # 2. Fit the Generalized Gaussian Distribution using Maximum Likelihood
    # scipy.stats.gennorm.fit returns: (shape/beta, location/mu, scale/alpha)
    beta, mu, alpha = scipy.stats.gennorm.fit(train_flat)
    
    # print(f"Fitted GGD: beta={beta:.4f}, mu={mu:.4f}, alpha={alpha:.4f}")
    
    # Ensure test data is a torch tensor for the rest of your pipeline
    if not isinstance(data_test, torch.Tensor):
        data_test = torch.tensor(data_test, dtype=torch.float32)
        
    # Move parameters to the same device as test data
    device = data_test.device
    beta_t = torch.tensor(beta, dtype=torch.float32, device=device)
    mu_t = torch.tensor(mu, dtype=torch.float32, device=device)
    alpha_t = torch.tensor(alpha, dtype=torch.float32, device=device)
    
    # 3. Apply the Analytical Score Function (The Nonlinearity)
    # g(x) = (beta / alpha) * |(x - mu) / alpha|^(beta - 1) * sgn(x - mu)
    
    # Standardize the input
    z = (data_test - mu_t) / alpha_t
    
    # Clamp the absolute value with a tiny epsilon. 
    # If beta < 1 (heavy-tailed like Laplace) and z is exactly 0, 
    # 0^(negative number) will cause NaNs/Infs without this.
    eps = 1e-8
    abs_z = torch.clamp(torch.abs(z), min=eps)
    
    # Calculate the score
    transformed_test_data = (beta_t / alpha_t) * (abs_z ** (beta_t - 1.0)) * torch.sign(z)
    
    return transformed_test_data

def clip_data(x,lim_upper,lim_lower):
    """
    Limiter function.
    """
    y = x.clone()
    y[y>lim_upper] = lim_upper
    y[y<lim_lower] = lim_lower
    return y

def H_multi_harmonic(K,psi0,dim_inp):
    """
    Observation matrix for multi-harmonic signal.
    """
    n = torch.arange(dim_inp).reshape(-1,1)
    wk = 2*torch.pi*psi0*torch.arange(1,K+1).reshape(1,-1)
    H = torch.zeros(dim_inp,2*K)
    H[:,:K] = torch.cos(wk*n)
    H[:,K:] = torch.sin(wk*n)
    return H

def init_weights(m):
    """
    Initialize weights of conv1d layers using Xavier uniform technique (appropriate for tanh activations).
    """
    if isinstance(m, torch.nn.Conv1d):
        torch.nn.init.xavier_uniform_(m.weight, gain=torch.nn.init.calculate_gain("tanh"))
        # m.bias.data.fill_(0.0)

def auroc(fpr,tpr,fpr_max=1):
    """
    Compute AUROC FPR and TPR.
    """
    delta_fpr = torch.diff(fpr[fpr<=fpr_max])
    return torch.sum(tpr[fpr<=fpr_max][:-1]*delta_fpr).cpu().numpy()

def calc_roc_statistics(t0, t1, bins=1000):
    """
    Computes ROC statistics using vectorized broadcasting.
    Much faster than a for-loop, identical logic to original.
    """
    upper_lim = max(t0.max(), t1.max())
    lower_lim = min(t0.min(), t1.min())
    
    # Reshape gamma to (bins, 1) so it broadcasts against t0 and t1
    gamma = torch.linspace(lower_lim - 1e-1, upper_lim + 1e-1, bins, device=t0.device).view(-1, 1)
    
    # Broadcast comparison: computes all thresholds at once
    # Resulting shape before sum is (bins, len(t)), sum collapses to (bins,)
    tp = (t1 > gamma).sum(dim=1).float()
    fp = (t0 > gamma).sum(dim=1).float()
    
    # Total counts are static, no need to calculate fn or tn
    pos_count = t1.numel()
    neg_count = t0.numel()
    
    tpr = tp / pos_count
    fpr = fp / neg_count
    
    return tpr, fpr

class StudentT_noise_generator:
    """
    Generator for student-t noise passed through linear filter.
    """
    def __init__(self,nu,filt_coeffs):
        self.m = torch.distributions.studentT.StudentT(df=nu)
        self.filt_coeffs = filt_coeffs
    def gen_noise_sequences(self,sequence_len,batch_size):
        u = self.m.sample((batch_size,sequence_len))
        w = torch.tensor(scipy.signal.lfilter(self.filt_coeffs[1],self.filt_coeffs[0],u),dtype=torch.float32)
        return w
    
class EarlyStopper:
    """
    Returns true if validation_loss doesn't improve for patience iterations. min_delta sets margin for counting an increase in loss.
    """
    def __init__(self, patience=1, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = float('inf')

    def early_stop(self, validation_loss):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False

def gen_list_models(config):
    """
    Returns list of dictionaries from dictionary of lists.
    """
    mesh = np.meshgrid(*config.values())
    model_list = [dict() for _ in range(len(mesh[0].flatten()))]
    for i in range(len(model_list)):
        for j,key in enumerate(config):
            model_list[i][key] = mesh[j].flatten()[i].copy()
    return model_list

class StandardClassifierCNN(torch.nn.Module):
    """
    A canonical, hierarchical CNN optimized for time-series classification.
    Uses temporal downsampling (MaxPool) and expanding channel dimensions.
    """
    def __init__(self, config):
        super().__init__()
        
        layers = []
        
        # --- Block 1: Extract local features & halve sequence length ---
        layers.append(torch.nn.Conv1d(1, 16, kernel_size=16, padding="same"))
        layers.append(torch.nn.ReLU())
        layers.append(torch.nn.MaxPool1d(kernel_size=2, stride=2))
        
        # --- Block 2: Expand features & halve sequence length again ---
        layers.append(torch.nn.Conv1d(16, 32, kernel_size=8, padding="same"))
        layers.append(torch.nn.ReLU())
        layers.append(torch.nn.MaxPool1d(kernel_size=2, stride=2))
        
        # --- Block 3: Deep features ---
        layers.append(torch.nn.Conv1d(32, 64, kernel_size=4, padding="same"))
        layers.append(torch.nn.ReLU())
        # We stop pooling here to avoid collapsing the sequence too early on small inputs
        
        self.feature_extractor = torch.nn.Sequential(*layers)
        
        # --- Classification Head ---
        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(64, 64),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.2), # Standard regularization for small datasets
            torch.nn.Linear(64, 1)
        )
        
        # Standard weight initialization
        for m in self.modules():
            if isinstance(m, torch.nn.Conv1d) or isinstance(m, torch.nn.Linear):
                torch.nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    torch.nn.init.zeros_(m.bias)

    def forward(self, x):
        # 1. Hierarchical Feature Extraction
        features = self.feature_extractor(x)
        
        # 2. Global Avg Pooling (adaptive to any remaining sequence length)
        # This retains the vital phase/shift invariance we discussed.
        pooled_features = torch.mean(features, dim=-1)#, _ 
        
        # 3. Final Classification
        logits = self.classifier(pooled_features)
        
        return logits

def generate_dynamic_batch(data_h0, H, config, fixed_amp=None):
    """
    Helper function to dynamically generate H1 data on the fly.
    If fixed_amp is None, samples amplitudes uniformly between min and max.
    """
    batch_size = data_h0.shape[0]
    device = data_h0.device
    num_harmonics = H.shape[1] // 2
    
    # 1. Random phases for each sample in the batch
    phases = 2 * torch.pi * torch.rand((batch_size, num_harmonics), dtype=torch.float32, device=device)
    
    # 2. Random or fixed amplitudes
    if fixed_amp is None:
        # Uniform sampling between [amp_min, amp_max]
        amp_min = config.get("amp_train_min", 0.05)
        amp_max = config.get("amp_train_max", 0.7)
        amps = (amp_max - amp_min) * torch.rand((batch_size, 1), dtype=torch.float32, device=device) + amp_min
    else:
        amps = torch.full((batch_size, 1), fixed_amp, dtype=torch.float32, device=device)
        
    # 3. Construct the parameter vector theta
    theta = torch.zeros((batch_size, H.shape[1]), dtype=torch.float32, device=device)
    theta[:, :num_harmonics] = amps * torch.cos(phases)
    theta[:, num_harmonics:] = -amps * torch.sin(phases)
    
    # 4. Generate the signal and add to noise (Assuming data is [batch, channels, seq_len])
    # H is [seq_len, num_harmonics*2]. We want signal of shape [batch, 1, seq_len]
    signal = torch.matmul(theta, H.T).unsqueeze(1)
    # print(data_h0.shape,signal.shape)
    data_h1 = data_h0 + signal
    data_h1 /= 1+torch.norm(signal, dim=(1, 2), keepdim=True) # Optional: Normalize to keep scale consistent, can help training stability
    
    # 5. Concatenate H0 and H1, and create labels
    x_batch = torch.cat([data_h0, data_h1], dim=0)
    y_batch = torch.cat([torch.zeros(batch_size, device=device), 
                         torch.ones(batch_size, device=device)], dim=0).unsqueeze(1)
                         
    return x_batch, y_batch


def train_nested_cv_val_bce(model, data_train, data_val, H, config):
    """
    Train BCE model until AUROC stopping condition is reached.
    """
    match(config["optim"]):
        case "SGD":
            optimizer = torch.optim.SGD(model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"])
        case "AdamW":
            optimizer = torch.optim.AdamW(model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"])
        case "Adam":
            optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"])
            
    criterion = torch.nn.BCEWithLogitsLoss()
    model.train()
    
    # auroc_val = []
    bce_val = []
    patience = 15
    best_bce = np.inf
    epochs_no_improve = 0
    
    for epoch in range(config["max_iterations"]):
        # Dynamic Batch Generation for Training
        x_train, y_train = generate_dynamic_batch(data_train, H, config)
        
        optimizer.zero_grad()
        logits_train = model(x_train)
        loss = criterion(logits_train, y_train)
        loss.backward()
        optimizer.step()
        
        # Validation using AUROC
        with torch.no_grad():
            model.eval()
            x_val, y_val = generate_dynamic_batch(data_val, H, config)
            logits_val = model(x_val)
            bce_val.append(criterion(logits_val, y_val).cpu().numpy())
            # probs_val = torch.sigmoid(logits_val).cpu().numpy()
            # y_val_np = y_val.cpu().numpy()
            
            # # Calculate AUROC
            # current_auroc = roc_auc_score(y_val_np, probs_val)
            # auroc_val.append(current_auroc)
            model.train()
            
        # Custom Early Stopping based on maximizing AUROC
        if epoch > 0: # Allow some initial epochs for learning before checking for improvements
            if bce_val[-1] < best_bce:
                best_bce = bce_val[-1]
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
                
            if epochs_no_improve >= patience:
                break
        # print(epoch, bce_val[-1])
            
    return min(bce_val), np.argmin(bce_val)


def train_nested_cv_test_bce(model, data_train, data_test, H, amplitudes, config, epochs):
    """
    Train BCE model for fixed epochs and test across different signal amplitudes.
    """
    match(config["optim"]):
        case "SGD":
            optimizer = torch.optim.SGD(model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"])
        case "AdamW":
            optimizer = torch.optim.AdamW(model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"])
        case "Adam":
            optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"])
            
    criterion = torch.nn.BCEWithLogitsLoss()
    model.train()
    
    loss_history = []
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        
    start_time = time.perf_counter()
    for i in range(epochs):
        x_train, y_train = generate_dynamic_batch(data_train, H, config)
        optimizer.zero_grad()
        logits_train = model(x_train)
        loss = criterion(logits_train, y_train)
        loss.backward()
        optimizer.step()
        loss_history.append(loss.item())
    
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        
    end_time = time.perf_counter()
    training_time = end_time - start_time
    model.eval()
    
    with torch.no_grad():
        # Evaluate H0 (Noise only)
        # We output probabilities (sigmoid) instead of raw logits so it acts like a test statistic [0, 1]
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            
        start_time = time.perf_counter()
        t0 = torch.sigmoid(model(data_test)).cpu().squeeze()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            
        end_time = time.perf_counter()
        inference_time = end_time - start_time
        t1 = [None] * len(amplitudes)
        
        # Evaluate H1 for each specific test amplitude
        for i in range(len(amplitudes)):
            # Generate H1 data with the fixed testing amplitude, but random phases
            x_test_h1, _ = generate_dynamic_batch(data_test, H, config, fixed_amp=amplitudes[i])
            
            # x_test_h1 contains both H0 and H1. We only want the H1 part (the second half)
            batch_size = data_test.shape[0]
            only_h1 = x_test_h1[batch_size:] 
            
            t1[i] = torch.sigmoid(model(only_h1)).cpu().squeeze()
            
    # Note: lambda_nc (non-centrality) is mathematically undefined for BCE classifiers, 
    # so we return a dummy zero tensor of the same shape to maintain return signature compatibility.
    dummy_lambda_nc = torch.zeros(len(amplitudes), dtype=torch.float32)
    
    return t0, torch.stack(t1, dim=0), dummy_lambda_nc, loss_history, training_time, inference_time


def nested_cv_bce(data, H, amplitudes, config_list, n_splits_o, n_repeats_o, n_splits_i, n_repeats_i, n_repeats_phase_net,train_fraction=None):
    """
    Nested cross validation procedure for the BCE CNN baseline.
    """
    if train_fraction==None:
        train_fraction = 1.0
    t0 = []
    t1 = []
    lambda_nc = []
    metric_history = []
    config_best = []
    epochs_best = []
    training_times, training_hyper_times, inference_times = [], [], []
    
    # rkf = RepeatedKFold(n_splits=n_splits_o, n_repeats=n_repeats_o)
    
    # for train_index, test_index in (pbar := tqdm(rkf.split(data), total=n_repeats_o * n_splits_o, leave=False)):
    for train_index, test_index in (pbar:=tqdm(customized_repeated_kfold(data, n_splits=n_splits_o, n_repeats=n_repeats_o, train_fraction=train_fraction),total=n_repeats_o*n_splits_o,leave=False)):
        pbar.set_description("outer CV (BCE)")
        auroc_val = []
        epochs_max_mean = []
        
        for config in (pbar_config := tqdm(config_list, leave=False)):
            pbar_config.set_description("hyper params")
            auroc_val_ = []
            epochs_max = []
            
            rkf_inner = RepeatedKFold(n_splits=n_splits_i, n_repeats=n_repeats_i)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                
            start_time = time.perf_counter()
            for train_index_i, val_index in (pbar_inner := tqdm(rkf_inner.split(data[train_index]), total=n_repeats_i * n_splits_i, leave=False)):
                pbar_inner.set_description("inner CV (BCE)")
                
                model = StandardClassifierCNN(config).to(device=data.device) # Using the StandardClassifierCNN from previous step
                
                auroc_max, epoch_max = train_nested_cv_val_bce(model, data[train_index_i], data[val_index], H, config)
                # print(epoch_max)
                auroc_val_.append(auroc_max)
                epochs_max.append(epoch_max)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                
            end_time = time.perf_counter()
            training_hyper_times.append(end_time - start_time)
            auroc_val.append(np.mean(auroc_val_))
            epochs_max_mean.append(round(np.mean(epochs_max)))
            
        config_best_ = config_list[np.argmax(auroc_val)]#
        # config_best_ = config_list[0]
        config_best.append(config_best_)
        epochs_best_ = max(1, epochs_max_mean[np.argmax(auroc_val)]) # Ensure at least 1 epoch
        # epochs_best_ = 200 # Using a fixed epoch count for testing phase to ensure consistency, can be tuned based on validation results
        epochs_best.append(epochs_best_)
        
        for i in range(n_repeats_phase_net):
            model = StandardClassifierCNN(config_best_).to(device=data.device)
            t0_, t1_, lambda_nc_, metric_, training_time, inference_time = train_nested_cv_test_bce(
                model, data[train_index], data[test_index], H, amplitudes, config_best_, epochs_best_)
                
            t0.append(t0_)
            t1.append(t1_)
            lambda_nc.append(lambda_nc_)
            metric_history.append(metric_)
            training_times.append(training_time)
            inference_times.append(inference_time)
    return torch.concat(t0), torch.concat(t1, dim=1), torch.mean(torch.stack(lambda_nc), dim=0), metric_history, config_best, training_times, training_hyper_times, inference_times
