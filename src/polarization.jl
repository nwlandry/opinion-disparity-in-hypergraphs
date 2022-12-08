using IntervalArithmetic, IntervalRootFinding, StaticArrays, LinearAlgebra

function get_polarization(epsilon2, epsilon3, beta2, beta3)
    psi_max = 0
    nu_max = Inf

    function f((x1, x2))
        dxdt1 = -x1 + 0.5*beta2*(1 - x1)*(x1 + x2 + epsilon2*(x1 - x2)) + 0.25*beta3*(1 - x1)*((x1 + x2)^2 + epsilon3*(3*x1^2 - 2*x1*x2 - x2^2))
        dxdt2 = -x2 + 0.5*beta2*(1 - x2)*(x2 + x1 + epsilon2*(x2 - x1)) + 0.25*beta3*(1 - x2)*((x2 + x1)^2 + epsilon3*(3*x2^2 - 2*x2*x1 - x1^2))
        return SVector(dxdt1, dxdt2)
    end

    function jacobian((x1, x2))
        J = zeros(2, 2)
        J[1, 1] = -1 + 0.5*beta2*(1 - x1)*(1 + epsilon2) - 0.5*beta2*(x1 + x2 + epsilon2*(x1 - x2)) +
                0.5*beta3*(1 - x1)*(x1 + x2 + epsilon3*(3*x1 - x2)) - 0.25*beta3*((x1 + x2)^2 + epsilon3*(3*x1^2 - 2*x1*x2 - x2^2))
        J[1, 2] = 0.5*beta2*(1 - x1)*(1 - epsilon2) + 0.5*beta3*(1 - x1)*(x1 + x2 - epsilon3*(x1 + x2))
        J[2, 1] = 0.5*beta2*(1 - x2)*(1 - epsilon2) + 0.5*beta3*(1 - x2)*(x2 + x1 - epsilon3*(x2 + x1))
        J[2, 2] = -1 + 0.5*beta2*(1 - x2)*(1 + epsilon2) - 0.5*beta2*(x2 + x1 + epsilon2*(x2 - x1)) +
                0.5*beta3*(1 - x2)*(x2 + x1 + epsilon3*(3*x2 - x1)) - 0.25*beta3*((x2 + x1)^2 + epsilon3*(3*x2^2 - 2*x2*x1 - x1^2))
        return J
    end

    function spectral_abscissa(J)
        return maximum(real(eigvals(J)))
    end
    
    result = roots(f, IntervalBox(0.0..0.5, 0.5..1.0))
    for rt in result
        x = mid(rt.interval)
        psi = abs(x[1] - x[2])


        if psi > 0.01 # filter out symmetric roots to save time
            nu = spectral_abscissa(jacobian(x))
            
            if (psi > psi_max && nu < 0)
                psi_max = psi
                nu_max = nu
            end
        end
    end

    return psi_max, nu_max
end


function get_imbalanced_polarization(k, q, rho, epsilon2, epsilon3, beta2, beta3)
    psi1_max = 0
    psi2_max = 0
    nu1_max = Inf
    nu2_max = Inf

    r2 = 1/(rho^2 + (1 - rho)^2) - 1
    r3 = 1/(rho^3 + (1 - rho)^3) - 1

    function f((x1, x2))
        dxdt1 = -x1 + beta2*k*(1 - x1)*(rho*(1 + r2*epsilon2)*x1 + (1 - rho)*(1 - epsilon2)*x2) +
            beta3*q*(1 - x1)*(rho^2*(1 + r3*epsilon3)*x1^2 + 2*rho*(1 - rho)*(1 - epsilon3)*x1*x2 +
            (1 - rho)^2*(1 - epsilon3)*x2^2)
        dxdt2 = -x2 + beta2*k*(1 - x2)*((1 - rho)*(1 + r2*epsilon2)*x2 + rho*(1 - epsilon2)*x1) +
            beta3*q*(1 - x2)*((1 - rho)^2*(1 + r3*epsilon3)*x2^2 + 2*(1 - rho)*rho*(1 - epsilon3)*x2*x1 +
            rho^2*(1 - epsilon3)*x1^2)
        return SVector(dxdt1, dxdt2)
    end

    function jacobian((x1, x2))
        J = zeros(2, 2)

        J[1, 1] = -1 + beta2*k*rho*(1 - x1)*(1 + r2*epsilon2) -
        beta2*k*((1 - rho)*(1 - epsilon2)*x2 + rho*(1 + r2*epsilon2)*x1) +
        2*beta3*q*(1 - x1)*(rho^2*(1 + r3*epsilon3)*x1 +
        (1 - rho)*rho*(1 - epsilon3)*x2) -
        beta3*q*(rho^2*(1 + r3*epsilon3)*x1^2 +
        2*(1 - rho)*rho*(1 - epsilon3)*x1*x2 + (1 - rho)^2*(1 - epsilon3)*x2^2)

        J[1, 2] = beta2*k*(1 - rho)*(1 - epsilon2)*(1 - x1) +
        2*beta3*q*(1 - rho)*rho*(1 - epsilon3)*(1 - x1)*x1 +
        2*beta3*q*(1 - rho)^2*(1 - epsilon3)*(1 - x1)*x2

        J[2, 1] = beta2*k*rho*(1 - epsilon2)*(1 - x2) +
        2*beta3*q*rho*(1 - rho)*(1 - epsilon3)*(1 - x2)*x2 +
        2*beta3*q*rho^2*(1 - epsilon3)*(1 - x2)*x1

        J[2, 2] = -1 + beta2*k*(1 - rho)*(1 - x2)*(1 + r2*epsilon2) -
        beta2*k*(rho*(1 - epsilon2)*x1 + (1 - rho)*(1 + r2*epsilon2)*x2) +
        2*beta3*q*(1 - x2)*((1 - rho)^2*(1 + r3*epsilon3)*x2 +
        rho*(1 - rho)*(1 - epsilon3)*x1) -
        beta3*q*((1 - rho)^2*(1 + r3*epsilon3)*x2^2 +
        2*rho*(1 - rho)*(1 - epsilon3)*x2*x1 + rho^2*(1 - epsilon3)*x1^2)
        
        return J
    end

    function spectral_abscissa(J)
        return maximum(real(eigvals(J)))
    end
    
    result = roots(f, IntervalBox(0..1, 0..1))
    for rt in result
        x = mid(rt.interval)
        nu = spectral_abscissa(jacobian(x))

        if (x[1] - x[2] > max(psi1_max, 0.01) && nu < 0)
            psi1_max = x[1] - x[2]
            nu1_max = nu
        elseif (x[2] - x[1] > max(psi2_max, 0.01) && nu < 0)
            psi2_max = x[2] - x[1]
            nu2_max = nu
        end
    end

    return psi1_max, nu1_max, psi2_max, nu2_max
end