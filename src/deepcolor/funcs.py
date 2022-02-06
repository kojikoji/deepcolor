import torch.distributions as dist


def calc_kld(qz):
    kld = -0.5 * (1 + qz.scale.pow(2).log() - qz.loc.pow(2) - qz.scale.pow(2))
    return(kld)


def calc_poisson_loss(ld, norm_mat, obs):
    p_z = dist.Poisson(ld * norm_mat)
    l = - p_z.log_prob(obs)
    return(l)
        
    
def calc_nb_loss(ld, norm_mat, theta, obs):
    ld = norm_mat * ld
    p =  ld / (ld + theta)
    p_z = dist.NegativeBinomial(theta, p)
    l = - p_z.log_prob(obs)
    return(l)
        
