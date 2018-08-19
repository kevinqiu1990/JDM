function [alpha_norm] = normalizeAlpha(alpha, type)

    if type == 1
        alpha_norm = (alpha-repmat(min(alpha),size(alpha,1),1))./(max(alpha)-min(alpha));
    else
        alpha_norm = zscore(alpha);
    end

end