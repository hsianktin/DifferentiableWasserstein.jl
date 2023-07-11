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

export W₁
end
