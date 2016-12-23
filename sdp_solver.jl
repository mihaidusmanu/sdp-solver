eps = 1e-6;
#C = [[1 2 3]; [2 9 0]; [3 0 7]];
#A = zeros(2, 3, 3);
#A[1, :, :] = [[1, 0, 1]; [0, 3, 7]; [1, 7, 5]];
#A[2, :, :] = [[0, 2, 8]; [2, 6, 0]; [8, 0, 4]];
#b = [11; 19];
# opt value 13.9
#C = [[1 0 0 0]; [0 1 0 0]; [0 0 2 0];[0 0 0 2]];
#A = zeros(1, 4, 4);
#A[1, :, :] = [[1 0 0 0]; [0 1 0 0]; [0 0 0 0]; [0 0 0 0]];
#b = [4];
# opt value 4

function sdp_solver(C, A, b)
  m = size(A, 1)
  n = size(A, 2)
  eye3d = zeros(1, n, n)
  eye3d[1, :, :] = eye(n)
  new_A = cat(1, A, eye3d)
  new_b = zeros(m + 1)
  new_b[m + 1] = 1
  mu = barrier_method_stop(C, new_A, new_b, [zeros(m); -minimum(eig(C)[1]) + 1])[2];
  if mu[m + 1] >= 0
    error("No strictly feasible point.")
  else
    barrier_method(C, A, b, mu[1 : m])
  end
end

function barrier_method(C, A, b, mu_0)
  n = size(A, 2)
  mu = mu_0
  gamma = 15
  t = 10
  while true
    mu = newton(t, C, A, b, mu)
    if n / t < eps
      break
    end
    t = gamma * t
  end
  (1 / t * inv(F(C, A, mu)), mu)
end

function barrier_method_stop(C, A, b, mu_0)
  m = size(A, 1)
  n = size(A, 2)
  mu = mu_0
  gamma = 15
  t = 1
  while true
    mu = newton_stop(t, C, A, b, mu)
    if n / t < eps || mu[m] < 0
      break
    end
    t = gamma * t
  end
  (1 / t * inv(F(C, A, mu)), mu)
end

function newton(t, C, A, b, x_0)
  x = x_0
  while true
    nabla_f_t_x = nabla_f_t(t, C, A, b, x)
    nabla2_f_t_x = nabla2_f_t(t, C, A, b, x)
    inv_nabla2_f_t_x = inv(nabla2_f_t_x)
    delta_x = - inv_nabla2_f_t_x * nabla_f_t_x
    lambda2 = - (nabla_f_t_x' * delta_x)[1]
    if lambda2 / 2 < eps
      break
    end
    fact = line_search(t, C, A, b, x, delta_x)
    x += fact * delta_x
  end
  x
end

function newton_stop(t, C, A, b, x_0)
  m = size(A, 1)
  x = x_0
  max_iter = 100
  iter = 0
  while true
    iter += 1
    nabla_f_t_x = nabla_f_t(t, C, A, b, x)
    nabla2_f_t_x = nabla2_f_t(t, C, A, b, x)
    inv_nabla2_f_t_x = inv(nabla2_f_t_x)
    delta_x = - inv_nabla2_f_t_x * nabla_f_t_x
    lambda2 = - (nabla_f_t_x' * delta_x)[1]
    if lambda2 / 2 < eps || x[m] < 0 || iter > max_iter
      break
    end
    fact = line_search(t, C, A, b, x, delta_x)
    x += fact * delta_x
  end
  if iter > max_iter
    x_0
  else
    x
  end
end

function line_search(t, C, A, b, x, delta_x)
  alpha = 0.01
  beta = 0.5
  fact = 1
  nabla_f_t_x = nabla_f_t(t, C, A, b, x)
  f_t_x = f_t(t, C, A, b, x)[1]
  while f_t(t, C, A, b, x + fact * delta_x)[1] > f_t_x + (alpha * fact * nabla_f_t_x' * delta_x)[1]
    fact = beta * fact
  end
  fact
end

function nabla2_f_t(t, C, A, b, mu)
  m = size(A, 1)
  n = size(A, 2)
  h = zeros(m, m)
  inv_F_mu = inv(F(C, A, mu))
  for i = 1 : m
    for j = 1 : m
      h[i, j] = trace(A[i, :, :]' * inv_F_mu * A[j, :, :] * inv_F_mu)
    end
  end
  h
end

function nabla_f_t(t, C, A, b, mu)
  m = size(A, 1)
  v = zeros(m)
  inv_F_mu = inv(F(C, A, mu))
  for i = 1 : m
    v[i] = trace(inv_F_mu * A[i, :, :])
  end
  t * b - v
end

function f_t(t, C, A, b, mu)
  F_mu = F(C, A, mu)
  if minimum(eig(F_mu)[1]) <= 0
    Inf
  else
    t * b' * mu - log(det(F_mu))
  end
end

function F(C, A, mu)
  m = size(A, 1)
  n = size(A, 2)
  aux = zeros(n, n)
  for i = 1 : m
    aux += mu[i] * A[i, :, :]
  end
  aux + C
end
