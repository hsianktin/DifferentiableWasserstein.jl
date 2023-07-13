module DifferentiableWasserstein
function W₁(u_samples, v_samples)
    # adapted from https://github.com/nklb/wasserstein-distance
    u_samples_sorted = sort(u_samples);
    v_samples_sorted = sort(v_samples);    
    all_samples = unique([u_samples_sorted; v_samples_sorted]) |> sort
    u_cdf = cdf(u_samples_sorted).(all_samples[1:end-1])
    v_cdf = cdf(v_samples_sorted).(all_samples[1:end-1])
    

    wsd = sum(abs.(u_cdf - v_cdf) .* diff(all_samples)); # 1-Wasserstein distance
    return wsd
end

cdf(u_samples) = x -> sum(u_samples .<= x) / length(u_samples)

function W₂(u_samples, v_samples)
    # adapted from https://github.com/nklb/wasserstein-distance
    u_samples_sorted = sort(u_samples);
    v_samples_sorted = sort(v_samples);    
    u_icdf_grids = [i / length(u_samples) for i in 0:length(u_samples)]
    v_icdf_grids = [i / length(v_samples) for i in 0:length(v_samples)]
    grids = unique([u_icdf_grids; v_icdf_grids]) |> sort
    U_icdf = quantile(u_samples).(grids[1:end-1])
    V_icdf = quantile(v_samples).(grids[1:end-1])
    return sqrt(sum((U_icdf - V_icdf).^2 .* diff(grids)))
end

function quantile(samples)
    # return a function that computes the quantile of a given sample
    samples_sorted = sort(samples)
    return p -> samples_sorted[floor(Int, p * length(samples_sorted))+1]
end
export W₁, W₂
end
