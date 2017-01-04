epsilon = 1e-6

function rank1_solution(C, A, M)
  n = size(M, 1)
  eigenvals = eig(M)[1]
  rk = 0
  for i = 1 : n
    if eigenvals[i] > epsilon
      rk = rk + 1
    end
  end
  if rk == 0
    # Rank = 0
    # 0 is a solution
    zeros(n)
  elseif rk == 1
    # Rank = 1
    # M is an orthogonal projector multiplied by a constant (the eigenvalue that is not null)
    aux = eigs(M)
    val = aux[1][1]
    v = sqrt(val) * aux[2][:, 1]
    v
  elseif rk == 2
    # Rank = 2
    aux = eigs(M)
    # V satisfies M = V * V'
    V = [aux[2][:, 1 : 2] zeros(n, n - 2)] * [[sqrt(aux[1][1]) 0]; [0 sqrt(aux[1][2])]; zeros(n - 2, 2)]
    (lambda, P) = eig(V' * C * V)
    V = V * P
    # We change V to have V' * C * V diagonal
    sigma_gamma = V' * A * V
    sigma = [sigma_gamma[1, 1]; sigma_gamma[2, 2]]
    gamma = sigma_gamma[1, 2]
    mat = [lambda'; sigma']
    if -epsilon < det(mat) < epsilon
      w = [sum(lambda); sum(sigma)]
      # (lambda_1, sigma_1), (lambda_2, sigma_2) isn't a base of the R^2
      if w[1] * lambda[1] > 0
        alpha = sqrt(w[1] / lambda[1])
        alpha * V[:, 1]
      else
        beta = sqrt(w[2] / lambda[2])
        beta * V[:, 2]
      end
    else
      z = inv(mat) * [0; gamma]
      # We solve t^2 + 2t(z_2 - z_1) = 1
      delta = 4 * (z[2] - z[1])^2 + 4
      roots = [(- 2 * (z[2] - z[1]) + sqrt(delta)) / 2; (- 2 * (z[2] - z[1]) - sqrt(delta)) / 2]
      t = 0
      # We pick the root r that satisfies 1 + 2 * r * z_1 > 0
      if 1 + 2 * roots[1] * z[1] > 0
        t = roots[1]
      else
        t = roots[2]
      end
      alpha = 1 / sqrt(1 + 2 * t * z[1])
      beta = t * alpha
      alpha * V[:, 1] + beta * V[:, 2]
    end
  else
    # Rank > 2
    # We pick a random eigenvector of M
    aux = eigs(M)
    val = aux[1][1]
    print(val)
    print("\n")
    v = sqrt(val) * aux[2][:, 1]
    # We look for a solution equivalent to M - v * v', new_v * new_v'
    new_v = rank1_solution(C, A, M - v * v')
    # We look for a solution equivalent to v * v' + new_v * new_v'
    rank1_solution(C, A, v * v' + new_v * new_v')
  end
end
