//g++ -std=c++17 -O2 -Wall -Wextra -pedantic lbfgs_pica_toy.cpp -o lbfgs_pica_toy



#include <algorithm>
#include <cmath>
#include <cstddef>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <random>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

using Vector = std::vector<double>;
using Matrix = std::vector<Vector>;

struct ObjectiveResult {
    double J{};
    Vector residual;
    Vector dcal;
};

struct GradientResult {
    double J{};
    Vector gradient;
    Vector residual;
    Vector dcal;
};

struct LBFGSPair {
    Vector s;
    Vector y;
    double rho{};
};

struct InversionResult {
    Vector model;
    Vector history;
};

// ------------------------------------------------------------
// 1. Basic vector utilities
// ------------------------------------------------------------

double dot(const Vector& a, const Vector& b) {
    if (a.size() != b.size()) {
        throw std::runtime_error("dot(): vector size mismatch");
    }
    double value = 0.0;
    for (std::size_t i = 0; i < a.size(); ++i) {
        value += a[i] * b[i];
    }
    return value;
}

double norm(const Vector& a) {
    return std::sqrt(dot(a, a));
}

Vector add(const Vector& a, const Vector& b) {
    if (a.size() != b.size()) {
        throw std::runtime_error("add(): vector size mismatch");
    }
    Vector out(a.size());
    for (std::size_t i = 0; i < a.size(); ++i) {
        out[i] = a[i] + b[i];
    }
    return out;
}

Vector subtract(const Vector& a, const Vector& b) {
    if (a.size() != b.size()) {
        throw std::runtime_error("subtract(): vector size mismatch");
    }
    Vector out(a.size());
    for (std::size_t i = 0; i < a.size(); ++i) {
        out[i] = a[i] - b[i];
    }
    return out;
}

Vector scale(const Vector& a, double c) {
    Vector out(a.size());
    for (std::size_t i = 0; i < a.size(); ++i) {
        out[i] = c * a[i];
    }
    return out;
}

Vector axpy(const Vector& x, const Vector& y, double alpha) {
    // Returns x + alpha*y
    if (x.size() != y.size()) {
        throw std::runtime_error("axpy(): vector size mismatch");
    }
    Vector out(x.size());
    for (std::size_t i = 0; i < x.size(); ++i) {
        out[i] = x[i] + alpha * y[i];
    }
    return out;
}

Vector matvec(const Matrix& A, const Vector& x) {
    if (A.empty() || A.front().size() != x.size()) {
        throw std::runtime_error("matvec(): matrix/vector size mismatch");
    }
    Vector y(A.size(), 0.0);
    for (std::size_t i = 0; i < A.size(); ++i) {
        for (std::size_t j = 0; j < x.size(); ++j) {
            y[i] += A[i][j] * x[j];
        }
    }
    return y;
}

Vector transpose_matvec(const Matrix& A, const Vector& x) {
    if (A.empty() || A.size() != x.size()) {
        throw std::runtime_error("transpose_matvec(): matrix/vector size mismatch");
    }
    Vector y(A.front().size(), 0.0);
    for (std::size_t i = 0; i < A.size(); ++i) {
        for (std::size_t j = 0; j < A[i].size(); ++j) {
            y[j] += A[i][j] * x[i];
        }
    }
    return y;
}

// ------------------------------------------------------------
// 2. Multiparameter model utilities
// ------------------------------------------------------------

Vector pack_model(const Vector& vp, const Vector& vs) {
    Vector m;
    m.reserve(vp.size() + vs.size());
    m.insert(m.end(), vp.begin(), vp.end());
    m.insert(m.end(), vs.begin(), vs.end());
    return m;
}

std::pair<Vector, Vector> unpack_model(const Vector& m, std::size_t n) {
    if (m.size() != 2 * n) {
        throw std::runtime_error("unpack_model(): expected model size 2*n");
    }
    Vector vp(m.begin(), m.begin() + static_cast<std::ptrdiff_t>(n));
    Vector vs(m.begin() + static_cast<std::ptrdiff_t>(n), m.end());
    return {vp, vs};
}

Vector apply_bounds(
    const Vector& m,
    std::size_t n,
    double vp_min = 1.5,
    double vp_max = 5.0,
    double vs_min = 0.5,
    double vs_max = 3.0
) {
    auto [vp, vs] = unpack_model(m, n);

    for (std::size_t i = 0; i < n; ++i) {
        vp[i] = std::clamp(vp[i], vp_min, vp_max);
        vs[i] = std::clamp(vs[i], vs_min, vs_max);
        vs[i] = std::min(vs[i], 0.95 * vp[i]);
    }

    return pack_model(vp, vs);
}

// ------------------------------------------------------------
// 3. Toy nonlinear forward problem
// ------------------------------------------------------------

Matrix build_smoothing_matrix(std::size_t n, double sigma = 3.0) {
    Matrix A(n, Vector(n, 0.0));

    for (std::size_t i = 0; i < n; ++i) {
        double row_sum = 0.0;
        for (std::size_t j = 0; j < n; ++j) {
            const double dx = static_cast<double>(j) - static_cast<double>(i);
            A[i][j] = std::exp(-0.5 * (dx / sigma) * (dx / sigma));
            row_sum += A[i][j];
        }
        for (double& value : A[i]) {
            value /= row_sum;
        }
    }

    return A;
}

Vector forward_model(const Vector& m, std::size_t n, const Matrix& A) {
    auto [vp, vs] = unpack_model(m, n);

    const Vector d1 = matvec(A, vp);
    const Vector d2 = matvec(A, vs);

    Vector vp_vs(n);
    Vector vp2(n);
    for (std::size_t i = 0; i < n; ++i) {
        vp_vs[i] = 0.15 * vp[i] * vs[i];
        vp2[i] = 0.05 * vp[i] * vp[i];
    }

    const Vector d3 = matvec(A, vp_vs);
    const Vector d4 = matvec(A, vp2);

    Vector data;
    data.reserve(4 * n);
    data.insert(data.end(), d1.begin(), d1.end());
    data.insert(data.end(), d2.begin(), d2.end());
    data.insert(data.end(), d3.begin(), d3.end());
    data.insert(data.end(), d4.begin(), d4.end());
    return data;
}

ObjectiveResult objective_and_residual(
    const Vector& m,
    const Vector& dobs,
    std::size_t n,
    const Matrix& A
) {
    Vector dcal = forward_model(m, n, A);
    Vector residual = subtract(dobs, dcal);
    const double J = 0.5 * dot(residual, residual);
    return {J, std::move(residual), std::move(dcal)};
}

// ------------------------------------------------------------
// 4. Analytical gradient for the toy problem
// ------------------------------------------------------------

GradientResult gradient(
    const Vector& m,
    const Vector& dobs,
    std::size_t n,
    const Matrix& A
) {
    auto [vp, vs] = unpack_model(m, n);
    ObjectiveResult obj = objective_and_residual(m, dobs, n, A);

    Vector r1(obj.residual.begin(), obj.residual.begin() + static_cast<std::ptrdiff_t>(n));
    Vector r2(obj.residual.begin() + static_cast<std::ptrdiff_t>(n),
              obj.residual.begin() + static_cast<std::ptrdiff_t>(2 * n));
    Vector r3(obj.residual.begin() + static_cast<std::ptrdiff_t>(2 * n),
              obj.residual.begin() + static_cast<std::ptrdiff_t>(3 * n));
    Vector r4(obj.residual.begin() + static_cast<std::ptrdiff_t>(3 * n), obj.residual.end());

    const Vector At_r1 = transpose_matvec(A, r1);
    const Vector At_r2 = transpose_matvec(A, r2);
    const Vector At_r3 = transpose_matvec(A, r3);
    const Vector At_r4 = transpose_matvec(A, r4);

    Vector g_vp(n);
    Vector g_vs(n);

    for (std::size_t i = 0; i < n; ++i) {
        g_vp[i] = -At_r1[i]
                  - (0.15 * vs[i]) * At_r3[i]
                  - (0.10 * vp[i]) * At_r4[i];

        g_vs[i] = -At_r2[i]
                  - (0.15 * vp[i]) * At_r3[i];
    }

    return {
        obj.J,
        pack_model(g_vp, g_vs),
        std::move(obj.residual),
        std::move(obj.dcal)
    };
}

// ------------------------------------------------------------
// 5. Gradient preprocessing
// ------------------------------------------------------------

Vector smooth_vector(const Vector& x, int radius = 2) {
    if (radius <= 0) {
        return x;
    }

    Vector out(x.size(), 0.0);
    const int width = 2 * radius + 1;

    for (std::size_t i = 0; i < x.size(); ++i) {
        double sum = 0.0;
        int count = 0;

        for (int offset = -radius; offset <= radius; ++offset) {
            const long j = static_cast<long>(i) + offset;
            if (j >= 0 && j < static_cast<long>(x.size())) {
                sum += x[static_cast<std::size_t>(j)];
                ++count;
            }
        }

        // Same conceptual role as a moving average. At boundaries,
        // only available samples are included.
        out[i] = sum / static_cast<double>(count > 0 ? count : width);
    }

    return out;
}

Vector preprocess_gradient(
    const Vector& g,
    std::size_t n,
    int smooth_radius = 2,
    double scale_vp = 1.0,
    double scale_vs = 1.0
) {
    auto [g_vp, g_vs] = unpack_model(g, n);
    g_vp = smooth_vector(g_vp, smooth_radius);
    g_vs = smooth_vector(g_vs, smooth_radius);

    for (std::size_t i = 0; i < n; ++i) {
        g_vp[i] *= scale_vp;
        g_vs[i] *= scale_vs;
    }

    return pack_model(g_vp, g_vs);
}

// ------------------------------------------------------------
// 6. L-BFGS two-loop recursion
// ------------------------------------------------------------

Vector lbfgs_two_loop(const Vector& g, const std::vector<LBFGSPair>& memory) {
    Vector q = g;
    std::vector<double> alphas;
    alphas.reserve(memory.size());

    // First loop: newest to oldest
    for (auto it = memory.rbegin(); it != memory.rend(); ++it) {
        const double alpha = it->rho * dot(it->s, q);
        q = axpy(q, it->y, -alpha);
        alphas.push_back(alpha);
    }

    double gamma = 1.0;
    if (!memory.empty()) {
        const LBFGSPair& last = memory.back();
        const double sy = dot(last.s, last.y);
        const double yy = dot(last.y, last.y);
        if (yy > 0.0) {
            gamma = sy / yy;
        }
    }

    Vector z = scale(q, gamma);

    // Second loop: oldest to newest
    for (std::size_t i = 0; i < memory.size(); ++i) {
        const LBFGSPair& pair = memory[i];
        const double alpha = alphas[memory.size() - 1 - i];
        const double beta = pair.rho * dot(pair.y, z);
        z = axpy(z, pair.s, alpha - beta);
    }

    return z; // H*g
}

void update_lbfgs_memory(
    std::vector<LBFGSPair>& memory,
    const Vector& s,
    const Vector& y,
    std::size_t max_memory = 5,
    double curvature_eps = 1e-10
) {
    const double sy = dot(s, y);
    const double threshold = curvature_eps * norm(s) * norm(y);

    if (sy <= threshold) {
        std::cout << "  Skipping L-BFGS pair: curvature condition failed.\n";
        return;
    }

    memory.push_back({s, y, 1.0 / sy});

    if (memory.size() > max_memory) {
        memory.erase(memory.begin());
    }
}

// ------------------------------------------------------------
// 7. Pica finite-difference step length
// ------------------------------------------------------------

double choose_epsilon(
    const Vector& m,
    const Vector& p,
    std::size_t n,
    double fraction = 0.01
) {
    auto [vp, vs] = unpack_model(m, n);
    auto [p_vp, p_vs] = unpack_model(p, n);

    double max_vp = 0.0;
    double max_vs = 0.0;
    double max_p_vp = 0.0;
    double max_p_vs = 0.0;

    for (std::size_t i = 0; i < n; ++i) {
        max_vp = std::max(max_vp, std::abs(vp[i]));
        max_vs = std::max(max_vs, std::abs(vs[i]));
        max_p_vp = std::max(max_p_vp, std::abs(p_vp[i]));
        max_p_vs = std::max(max_p_vs, std::abs(p_vs[i]));
    }

    double epsilon = std::numeric_limits<double>::infinity();

    if (max_p_vp > 0.0) {
        epsilon = std::min(epsilon, fraction * max_vp / max_p_vp);
    }
    if (max_p_vs > 0.0) {
        epsilon = std::min(epsilon, fraction * max_vs / max_p_vs);
    }

    if (!std::isfinite(epsilon)) {
        return 1.0;
    }

    return epsilon;
}

std::pair<double, double> pica_step_length(
    const Vector& m,
    const Vector& p,
    std::size_t n,
    const Matrix& A,
    const Vector& dcal,
    const Vector& residual
) {
    const double epsilon = choose_epsilon(m, p, n, 0.01);

    Vector m_eps = apply_bounds(axpy(m, p, epsilon), n);
    const Vector d_eps = forward_model(m_eps, n, A);
    const Vector delta_d = subtract(d_eps, dcal);

    const double denominator = dot(delta_d, delta_d);
    if (denominator <= 0.0) {
        return {0.0, epsilon};
    }

    const double alpha = epsilon * dot(delta_d, residual) / denominator;
    return {alpha, epsilon};
}

// ------------------------------------------------------------
// 8. Main inversion loop: L-BFGS + Pica + backtracking
// ------------------------------------------------------------

InversionResult invert_lbfgs_pica(
    const Vector& m0,
    const Vector& dobs,
    std::size_t n,
    const Matrix& A,
    int niter = 30,
    std::size_t max_memory = 5,
    int max_backtracking = 8,
    int smooth_radius = 2
) {
    Vector m = m0;
    std::vector<LBFGSPair> memory;

    Vector m_prev;
    Vector g_prev;
    bool have_previous = false;

    Vector history;

    for (int iteration = 0; iteration < niter; ++iteration) {
        std::cout << "\nIteration " << iteration << "\n";

        GradientResult result = gradient(m, dobs, n, A);
        Vector g = preprocess_gradient(
            result.gradient,
            n,
            smooth_radius,
            1.0,
            1.0
        );

        std::cout << "  Misfit J = " << std::scientific << result.J << "\n";
        history.push_back(result.J);

        if (have_previous) {
            const Vector s = subtract(m, m_prev);
            const Vector y = subtract(g, g_prev);
            update_lbfgs_memory(memory, s, y, max_memory, 1e-10);
        }

        Vector p;
        if (memory.empty()) {
            p = scale(g, -1.0);
            std::cout << "  Direction: steepest descent\n";
        } else {
            const Vector Hg = lbfgs_two_loop(g, memory);
            p = scale(Hg, -1.0);
            std::cout << "  Direction: L-BFGS with "
                      << memory.size() << " stored pairs\n";
        }

        double gtp = dot(g, p);
        if (gtp >= 0.0) {
            std::cout << "  Direction is not descent. Resetting memory.\n";
            memory.clear();
            p = scale(g, -1.0);
            gtp = dot(g, p);
        }

        std::cout << "  g^T p = " << gtp << "\n";

        auto [alpha, epsilon] = pica_step_length(
            m,
            p,
            n,
            A,
            result.dcal,
            result.residual
        );

        std::cout << "  Pica epsilon = " << epsilon << "\n";
        std::cout << "  Pica alpha   = " << alpha << "\n";

        if (alpha <= 0.0 || !std::isfinite(alpha)) {
            std::cout << "  Non-positive/invalid alpha from Pica. "
                         "Using small fallback step.\n";
            alpha = 1e-3 / std::max(norm(p), 1e-12);
        }

        bool accepted = false;
        double alpha_trial = alpha;
        Vector m_trial;

        for (int bt = 0; bt <= max_backtracking; ++bt) {
            m_trial = apply_bounds(axpy(m, p, alpha_trial), n);
            const ObjectiveResult trial = objective_and_residual(m_trial, dobs, n, A);

            if (trial.J < result.J) {
                accepted = true;
                std::cout << "  Accepted alpha = " << alpha_trial
                          << ", J_trial = " << trial.J
                          << ", backtracking = " << bt << "\n";
                break;
            }

            alpha_trial *= 0.5;
        }

        if (!accepted) {
            std::cout << "  No acceptable step found. "
                         "Resetting memory and stopping.\n";
            memory.clear();
            break;
        }

        m_prev = m;
        g_prev = g;
        have_previous = true;
        m = std::move(m_trial);
    }

    return {std::move(m), std::move(history)};
}

// ------------------------------------------------------------
// 9. CSV output helpers
// ------------------------------------------------------------

void write_history_csv(const std::string& filename, const Vector& history) {
    std::ofstream file(filename);
    if (!file) {
        throw std::runtime_error("Cannot open " + filename);
    }

    file << "iteration,objective\n";
    for (std::size_t i = 0; i < history.size(); ++i) {
        file << i << ',' << std::setprecision(16) << history[i] << '\n';
    }
}

void write_models_csv(
    const std::string& filename,
    const Vector& x,
    const Vector& vp_true,
    const Vector& vp0,
    const Vector& vp_inv,
    const Vector& vs_true,
    const Vector& vs0,
    const Vector& vs_inv
) {
    std::ofstream file(filename);
    if (!file) {
        throw std::runtime_error("Cannot open " + filename);
    }

    file << "x,vp_true,vp_initial,vp_inverted,"
            "vs_true,vs_initial,vs_inverted\n";

    for (std::size_t i = 0; i < x.size(); ++i) {
        file << std::setprecision(16)
             << x[i] << ','
             << vp_true[i] << ','
             << vp0[i] << ','
             << vp_inv[i] << ','
             << vs_true[i] << ','
             << vs0[i] << ','
             << vs_inv[i] << '\n';
    }
}

// ------------------------------------------------------------
// 10. Complete runnable example
// ------------------------------------------------------------

int main() {
    try {
        constexpr std::size_t n = 120;
        Vector x(n);
        for (std::size_t i = 0; i < n; ++i) {
            x[i] = static_cast<double>(i) / static_cast<double>(n - 1);
        }

        const Matrix A = build_smoothing_matrix(n, 3.0);

        Vector vp_true(n);
        Vector vs_true(n);
        Vector vp0(n);
        Vector vs0(n);

        for (std::size_t i = 0; i < n; ++i) {
            const double xi = x[i];

            vp_true[i] = 2.2
                + 0.7 * std::exp(-std::pow((xi - 0.35) / 0.08, 2.0))
                + 0.5 * std::exp(-std::pow((xi - 0.75) / 0.10, 2.0));

            vs_true[i] = 1.1
                + 0.35 * std::exp(-std::pow((xi - 0.40) / 0.10, 2.0))
                + 0.25 * std::exp(-std::pow((xi - 0.70) / 0.12, 2.0));

            vp0[i] = 2.0 + 0.1 * xi;
            vs0[i] = 1.0 + 0.05 * xi;
        }

        const Vector m_true = pack_model(vp_true, vs_true);
        Vector dobs = forward_model(m_true, n, A);

        // Add deterministic Gaussian noise.
        double mean = 0.0;
        for (double value : dobs) {
            mean += value;
        }
        mean /= static_cast<double>(dobs.size());

        double variance = 0.0;
        for (double value : dobs) {
            variance += (value - mean) * (value - mean);
        }
        variance /= static_cast<double>(dobs.size());
        const double noise_std = 0.01 * std::sqrt(variance);

        std::mt19937 generator(7);
        std::normal_distribution<double> normal(0.0, noise_std);
        for (double& value : dobs) {
            value += normal(generator);
        }

        const Vector m0 = pack_model(vp0, vs0);

        InversionResult inversion = invert_lbfgs_pica(
            m0,
            dobs,
            n,
            A,
            40,  // iterations
            7,   // L-BFGS memory
            8,   // maximum backtracking steps
            1    // gradient smoothing radius
        );

        auto [vp_inv, vs_inv] = unpack_model(inversion.model, n);

        write_history_csv("lbfgs_history.csv", inversion.history);
        write_models_csv(
            "lbfgs_models.csv",
            x,
            vp_true,
            vp0,
            vp_inv,
            vs_true,
            vs0,
            vs_inv
        );

        std::cout << "\nFinished. Wrote:\n"
                  << "  lbfgs_history.csv\n"
                  << "  lbfgs_models.csv\n";

        return 0;
    } catch (const std::exception& error) {
        std::cerr << "Error: " << error.what() << '\n';
        return 1;
    }
}
