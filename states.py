from mamba_ssm.ops.triton.selective_state_update import selective_state_update
import torch 
from einops import rearrange, repeat
import torch.nn.functional as F

# ngroups = 2
# d_ssm = 8
# n_heads = 2
# d_state = 4
# headdim = d_ssm//n_heads
ngroups = 2
d_ssm = 8192
n_heads = 64
d_state = 128
headdim = d_ssm//n_heads

torch.manual_seed(0)
ssm_state = torch.randn(1,n_heads,headdim,d_state)
x = torch.randn(1, d_ssm)
dt = torch.randn(1,n_heads)
A = torch.randn(n_heads)
B = torch.randn(1,ngroups*d_state)
C = torch.randn(1,ngroups*d_state)
dt_bias = torch.randn(n_heads)
D = torch.randn(n_heads)

# print("ssm_state 1", ssm_state[0,0,0,:5])


def update_ssm_state(ssm_state, x, dt, A, B, C, dt_bias, D):

    dt = F.softplus(dt + dt_bias.to(dtype=dt.dtype))  # (batch, nheads)
    dA = torch.exp(dt * A)  # (batch, nheads)
    x = rearrange(x, "b (h p) -> b h p", p=headdim)

    # Step 1: Multiply dt and x over dimensions b and h
    intermediate = dt.unsqueeze(-1) * x  # Shape: (b, h, p)

    # Step 2: Expand intermediate and B to align dimensions for broadcasting
    intermediate = intermediate.unsqueeze(-1)  # Shape: (b, h, p, 1)
    B_expanded = B.unsqueeze(1).unsqueeze(1)   # Shape: (b, 1, 1, n)

    # Step 3: Perform element-wise multiplication to get dBx
    dBx = intermediate * B_expanded # Shape: (b, h, p, n)

    # dBx = torch.einsum("bh,bn,bhp->bhpn", dt, B, x)
    # Step 4: all reduce dBx over the last dimension, leaving head_dim/n_groups as dim
    # Reshape the array
    dBx = rearrange(dBx, "b h p (g n) -> b h p g n", g = ngroups)

    # print("dBx", dBx[0,0,0,:5,:5])

    # Sum over the n_groups dimension
    # dBx = dBx.sum(axis=-2)
    dBx = dBx[:,:,:,0,:]

    # print("forget", (ssm_state * rearrange(dA, "b h -> b h 1 1"))[0,0,:5,:5])

    ssm_state_custom = ssm_state * rearrange(dA, "b h -> b h 1 1") + dBx

    print("After update 1 ssm", ssm_state_custom[0,0,0,:5])

    return ssm_state_custom

ssm_state = update_ssm_state(ssm_state, x, dt, A, B, C, dt_bias, D)
_ = update_ssm_state(ssm_state, x, dt, A, B, C, dt_bias, D)

####################################3

torch.manual_seed(0)
ssm_state = torch.randn(1,n_heads,headdim,d_state)
x = torch.randn(1, d_ssm)
dt = torch.randn(1,n_heads)
A = torch.randn(n_heads)
B = torch.randn(1,ngroups*d_state)
C = torch.randn(1,ngroups*d_state)
dt_bias = torch.randn(n_heads)
D = torch.randn(n_heads)

# print("ssm_state 2", ssm_state[0,0,0,:5])


def update_ssm_state_cuda(ssm_state, x, dt, A, B, C, dt_bias, D):
    dt_x = F.softplus(dt + dt_bias.to(dtype=dt.dtype))  # (batch, nheads)
    dA = torch.exp(dt_x * A)  # (batch, nheads)

    A = repeat(A, "h -> h p n", p=headdim, n=d_state).to(device="cuda")
    dt = repeat(dt, "b h -> b h p", p=headdim)
    dt_bias = repeat(dt_bias, "h -> h p", p=headdim)
    D = repeat(D, "h -> h p", p=headdim)
    B = rearrange(B, "b (g n) -> b g n", g=ngroups)
    C = rearrange(C, "b (g n) -> b g n", g=ngroups)
    x_reshaped = rearrange(x, "b (h p) -> b h p", p=headdim)

    # send all to gpu
    A = A.to(device="cuda")
    dt = dt.to(device="cuda")
    dt_bias = dt_bias.to(device="cuda")
    D = D.to(device="cuda")
    B = B.to(device="cuda")
    C = C.to(device="cuda")
    x_reshaped = x_reshaped.to(device="cuda")
    ssm_state = ssm_state.to(device="cuda")

    previous_state = ssm_state.clone().to(device="cuda")

    y = selective_state_update(
                    ssm_state, x_reshaped, dt, A, B, C, D, z=None,
                    dt_bias=dt_bias, dt_softplus=True
                )
    # print(x.device.index)
    dBx2 = ssm_state - previous_state * rearrange(dA.to("cuda"), "b h -> b h 1 1")
    print("After update 2 ssm", ssm_state[0,0,0,:5])
    # print("dBx 2", dBx2[0,0,0,:5])

    return ssm_state

ssm_state = update_ssm_state_cuda(ssm_state, x, dt, A, B, C, dt_bias, D)
_ = update_ssm_state_cuda(ssm_state, x, dt, A, B, C, dt_bias, D)