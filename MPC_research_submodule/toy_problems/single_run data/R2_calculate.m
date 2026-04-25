function R2 = R2_calculate(y, y_hat)

% Calculate the mean of the observed data
y_mean = mean(y);

% Compute the sum of squares of residuals (SSR)
SSR = sum((y - y_hat).^2);

% Compute the total sum of squares (SST)
SST = sum((y - y_mean).^2);

% Calculate the R^2 value
R2 = 1 - (SSR / SST);


