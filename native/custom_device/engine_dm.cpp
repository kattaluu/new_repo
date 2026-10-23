#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/complex.h>
#include <pybind11/stl.h>

#include <algorithm>
#include <array>
#include <vector>
#include <complex>
#include <cstdint>
#include <random>
#include <stdexcept>
#include <cmath>

#ifndef NUM_QUBITS
#define NUM_QUBITS 4
#endif

#ifndef MODULE_NAME
#define MODULE_NAME engine_dm
#endif

namespace py = pybind11;
using cplx = std::complex<double>;

static constexpr int N   = NUM_QUBITS;
static constexpr int DIM = 1 << N;

// PennyLane-конвенция: wire=0 — ЛЕВЫЙ кубит в записи |q0 q1 ... q_{N-1}>,
// а внутренний индекс состояния i хранится как двоичное число с младшим битом справа.
// Поэтому позиция бита для wire: bitpos = N-1-wire.
inline int bitpos_for_wire(int wire) {
    return N - 1 - wire;
}

class Engine {
public:
    Engine()
    : rho_(DIM * DIM, cplx(0.0, 0.0)), rng_(std::random_device{}())
    {
        reset();
    }

    // ---------------- Базовые операции над состоянием -----------------

    // reset -> |0...0><0...0|
    void reset() {
        std::fill(rho_.begin(), rho_.end(), cplx(0.0, 0.0));
        rho_[0] = cplx(1.0, 0.0);
    }

    // |index><index|
    void set_basis_state(int index) {
        if (index < 0 || index >= DIM) {
            throw std::runtime_error("basis index out of range");
        }
        std::fill(rho_.begin(), rho_.end(), cplx(0.0, 0.0));
        rho_[index * DIM + index] = cplx(1.0, 0.0);
    }

    // полная матрица плотности (DIM x DIM)
    py::array_t<cplx> get_state() const {
        py::array_t<cplx> out({DIM, DIM});
        auto buf = out.mutable_unchecked<2>();
        for (int i = 0; i < DIM; ++i) {
            for (int j = 0; j < DIM; ++j) {
                buf(i, j) = rho_[i * DIM + j];
            }
        }
        return out;
    }

    // p(i) = rho_ii
    py::array_t<double> get_probs() const {
        py::array_t<double> out(DIM);
        auto buf = out.mutable_unchecked<1>();
        for (int i = 0; i < DIM; ++i) {
            buf(i) = rho_[i * DIM + i].real();
        }
        return out;
    }

    // маргинальные вероятности по подмножеству проводов
    py::array_t<double> get_probs_marginal(const std::vector<int> &wires) const {
        if (wires.empty()) {
            py::array_t<double> out(1);
            auto buf = out.mutable_unchecked<1>();
            buf(0) = 1.0;
            return out;
        }

        const int nw = static_cast<int>(wires.size());
        const int dim_marg = 1 << nw;

        py::array_t<double> out(dim_marg);
        auto buf = out.mutable_unchecked<1>();
        for (int i = 0; i < dim_marg; ++i) {
            buf(i) = 0.0;
        }

        // пробегаем все базисные состояния и собираем биты нужных wires
        for (int i = 0; i < DIM; ++i) {
            double p = rho_[i * DIM + i].real();
            if (p <= 0.0) {
                continue;
            }
            int idx = 0;
            for (int k = 0; k < nw; ++k) {
                int w = wires[k];
                if (w < 0 || w >= N) {
                    throw std::runtime_error("wire index out of range in get_probs_marginal");
                }
                int bp  = bitpos_for_wire(w);
                int bit = (i >> bp) & 1;
                idx = (idx << 1) | bit;
            }
            buf(idx) += p;
        }

        return out;
    }

    // ---------------- Сэмплирование -----------------

    py::array_t<std::uint8_t> generate_samples(int shots) {
        if (shots <= 0) {
            throw std::runtime_error("shots must be > 0");
        }

        std::vector<double> probs(DIM);
        double norm = 0.0;
        for (int i = 0; i < DIM; ++i) {
            double p = std::max(rho_[i * DIM + i].real(), 0.0);
            probs[i] = p;
            norm += p;
        }
        if (norm <= 0.0) {
            throw std::runtime_error("density matrix has non-positive trace");
        }
        for (int i = 0; i < DIM; ++i) {
            probs[i] /= norm;
        }

        std::discrete_distribution<int> dist(probs.begin(), probs.end());

        // результат (shots, N), j=0 соответствует wire=0 (левый кубит)
        py::array_t<std::uint8_t> out({shots, N});
        auto buf = out.mutable_unchecked<2>();

        for (int s = 0; s < shots; ++s) {
            int idx = dist(rng_);
            // декодируем индекс в биты q0..q_{N-1}
            for (int j = 0; j < N; ++j) {
                int bp  = N - 1 - j;                // wire j -> bitpos N-1-j
                std::uint8_t bit = (idx >> bp) & 1;
                buf(s, j) = bit;
            }
        }

        return out;
    }

    // histogram по подмножеству wires из уже готовой матрицы samples
    py::array_t<long long> count_states(py::array_t<std::uint8_t> samples,
                                        const std::vector<int> &wires,
                                        int num_states) const {
        auto sm = samples.unchecked<2>();
        int shots  = static_cast<int>(sm.shape(0));
        int ncols  = static_cast<int>(sm.shape(1));
        int nw     = static_cast<int>(wires.size());

        if (nw == 0) {
            py::array_t<long long> out(1);
            auto hist = out.mutable_unchecked<1>();
            hist(0) = shots;
            return out;
        }

        int m = 1 << nw;
        if (num_states > 0 && num_states != m) {
            throw std::runtime_error("num_states != 2^{|wires|} in count_states");
        }

        for (int w : wires) {
            if (w < 0 || w >= ncols) {
                throw std::runtime_error("wire index out of range in count_states(samples)");
            }
        }

        py::array_t<long long> out(m);
        auto hist = out.mutable_unchecked<1>();
        for (int i = 0; i < m; ++i) {
            hist(i) = 0;
        }

        for (int s = 0; s < shots; ++s) {
            int idx = 0;
            for (int k = 0; k < nw; ++k) {
                int w = wires[k];
                std::uint8_t bit = sm(s, w);   // здесь w уже в "пространстве wires"
                idx = (idx << 1) | bit;
            }
            hist(idx) += 1;
        }

        return out;
    }

    // ---------------- Унитарные операции -----------------

    void apply_one_qubit_unitary(const cplx U[4], int wire) {
        if (wire < 0 || wire >= N) {
            throw std::runtime_error("wire out of range in apply_one_qubit_unitary");
        }
        int bp   = bitpos_for_wire(wire);
        int mask = 1 << bp;

        std::vector<cplx> new_rho(DIM * DIM);
        std::fill(new_rho.begin(), new_rho.end(), cplx(0.0, 0.0));

        for (int i = 0; i < DIM; ++i) {
            int bi = (i >> bp) & 1;
            int i0 = i & ~mask;
            int i1 = i0 | mask;

            for (int j = 0; j < DIM; ++j) {
                int bj = (j >> bp) & 1;
                int j0 = j & ~mask;
                int j1 = j0 | mask;

                cplx sum(0.0, 0.0);

                for (int a = 0; a < 2; ++a) {
                    int ia = (a == 0) ? i0 : i1;
                    cplx u_left = U[bi * 2 + a];

                    for (int b = 0; b < 2; ++b) {
                        int jb = (b == 0) ? j0 : j1;
                        cplx u_right = std::conj(U[bj * 2 + b]);
                        sum += u_left * rho_[ia * DIM + jb] * u_right;
                    }
                }

                new_rho[i * DIM + j] = sum;
            }
        }

        rho_ = new_rho;
    }

    void apply_two_qubit_unitary(const cplx U[16], int wire0, int wire1) {
        if (wire0 < 0 || wire0 >= N || wire1 < 0 || wire1 >= N) {
            throw std::runtime_error("wire index out of range in apply_two_qubit_unitary");
        }
        if (wire0 == wire1) {
            throw std::runtime_error("wire0 == wire1 in apply_two_qubit_unitary");
        }

        int b0    = bitpos_for_wire(wire0);
        int b1    = bitpos_for_wire(wire1);
        int mask0 = 1 << b0;
        int mask1 = 1 << b1;

        std::vector<cplx> new_rho(DIM * DIM);
        std::fill(new_rho.begin(), new_rho.end(), cplx(0.0, 0.0));

        for (int i = 0; i < DIM; ++i) {
            int bi0 = (i >> b0) & 1;
            int bi1 = (i >> b1) & 1;
            int li  = (bi0 << 1) | bi1;   // локальный индекс (wire0, wire1)

            int env_i = i & ~(mask0 | mask1);

            for (int j = 0; j < DIM; ++j) {
                int bj0 = (j >> b0) & 1;
                int bj1 = (j >> b1) & 1;
                int lj  = (bj0 << 1) | bj1;

                int env_j = j & ~(mask0 | mask1);

                cplx sum(0.0, 0.0);

                for (int a = 0; a < 4; ++a) {
                    int a0 = (a >> 1) & 1;
                    int a1 = a & 1;
                    int ia = env_i | (a0 << b0) | (a1 << b1);
                    cplx u_left = U[li * 4 + a];

                    for (int b = 0; b < 4; ++b) {
                        int b0v = (b >> 1) & 1;
                        int b1v = b & 1;
                        int jb = env_j | (b0v << b0) | (b1v << b1);
                        cplx u_right = std::conj(U[lj * 4 + b]);

                        sum += u_left * rho_[ia * DIM + jb] * u_right;
                    }
                }

                new_rho[i * DIM + j] = sum;
            }
        }

        rho_ = new_rho;
    }

    // RX/RY/RZ/PhaseShift/CNOT/CRX/CPHASE

    void apply_rx(double theta, int wire) {
        double c = std::cos(theta / 2.0);
        double s = std::sin(theta / 2.0);
        cplx U[4] = {
            cplx(c, 0.0), cplx(0.0, -s),
            cplx(0.0, -s), cplx(c, 0.0)
        };
        apply_one_qubit_unitary(U, wire);
    }

    void apply_ry(double theta, int wire) {
        double c = std::cos(theta / 2.0);
        double s = std::sin(theta / 2.0);
        cplx U[4] = {
            cplx(c, 0.0), cplx(-s, 0.0),
            cplx(s, 0.0), cplx(c, 0.0)
        };
        apply_one_qubit_unitary(U, wire);
    }

    void apply_rz(double theta, int wire) {
        double half = theta / 2.0;
        cplx e_m = std::exp(cplx(0.0, -half));
        cplx e_p = std::exp(cplx(0.0, +half));
        cplx U[4] = {
            e_m, cplx(0.0, 0.0),
            cplx(0.0, 0.0), e_p
        };
        apply_one_qubit_unitary(U, wire);
    }

    void apply_phase_shift(double phi, int wire) {
        cplx phase = std::exp(cplx(0.0, phi));
        cplx U[4] = {
            cplx(1.0, 0.0), cplx(0.0, 0.0),
            cplx(0.0, 0.0), phase
        };
        apply_one_qubit_unitary(U, wire);
    }

    // Berkeley gate (PennyLane: qml.Berkeley / B_gate)
    // Матрица (в стандартном базисе |00>,|01>,|10>,|11|):
    // [[c1, 0,  0,  i s1],
    //  [0,  c3, i s3, 0 ],
    //  [0,  i s3, c3, 0 ],
    //  [i s1, 0,  0,  c1]]
    void apply_berkeley(int wire0, int wire1) {
        constexpr double pi = 3.141592653589793238462643383279502884;

        double c1 = std::cos(pi / 8.0);
        double c3 = std::cos(3.0 * pi / 8.0);
        cplx s1(0.0, std::sin(pi / 8.0));
        cplx s3(0.0, std::sin(3.0 * pi / 8.0));

        cplx U[16] = {};
        U[0 * 4 + 0] = cplx(c1, 0.0);
        U[0 * 4 + 3] = s1;

        U[1 * 4 + 1] = cplx(c3, 0.0);
        U[1 * 4 + 2] = s3;

        U[2 * 4 + 1] = s3;
        U[2 * 4 + 2] = cplx(c3, 0.0);

        U[3 * 4 + 0] = s1;
        U[3 * 4 + 3] = cplx(c1, 0.0);

        apply_two_qubit_unitary(U, wire0, wire1);
    }

    void apply_cnot(int control, int target) {
        cplx U[16] = {};
        U[0 * 4 + 0] = cplx(1.0, 0.0); // |00> -> |00|
        U[1 * 4 + 1] = cplx(1.0, 0.0); // |01> -> |01|
        U[2 * 4 + 3] = cplx(1.0, 0.0); // |10> -> |11|
        U[3 * 4 + 2] = cplx(1.0, 0.0); // |11> -> |10|
        apply_two_qubit_unitary(U, control, target);
    }

    void apply_crx(double theta, int control, int target) {
        double c = std::cos(theta / 2.0);
        double s = std::sin(theta / 2.0);
        cplx minus_i(0.0, -1.0);

        cplx U[16] = {};
        // |00>, |01> неизменны
        U[0 * 4 + 0] = cplx(1.0, 0.0);
        U[1 * 4 + 1] = cplx(1.0, 0.0);

        // подпространство |10>, |11|
        U[2 * 4 + 2] = c;
        U[2 * 4 + 3] = minus_i * s;
        U[3 * 4 + 2] = minus_i * s;
        U[3 * 4 + 3] = c;

        apply_two_qubit_unitary(U, control, target);
    }

    void apply_cphase(double phi, int control, int target) {
        cplx phase = std::exp(cplx(0.0, phi));
        cplx U[16] = {};
        U[0 * 4 + 0] = cplx(1.0, 0.0); // |00>
        U[1 * 4 + 1] = cplx(1.0, 0.0); // |01>
        U[2 * 4 + 2] = cplx(1.0, 0.0); // |10>
        U[3 * 4 + 3] = phase;          // |11>
        apply_two_qubit_unitary(U, control, target);
    }

    // U: 2x2 (один кубит) или DIMxDIM (полный оператор)
    void apply_unitary(py::array_t<cplx> U_in, const std::vector<int> &wires) {
        const int nw = static_cast<int>(wires.size());
        if (nw > N) throw std::runtime_error("too many wires");
        const int kdim = 1 << nw;
        auto Um = U_in.unchecked<2>();
        if (Um.shape(0) != kdim || Um.shape(1) != kdim)
            throw std::runtime_error("unitary shape does not match wires");
        int mask = 0;
        for (int wire : wires) {
            if (wire < 0 || wire >= N) throw std::runtime_error("wire out of range");
            int bit = 1 << bitpos_for_wire(wire);
            if (mask & bit) throw std::runtime_error("duplicate wire");
            mask |= bit;
        }
        if (nw == 1) {
            cplx U[4];
            for (int i = 0; i < 2; ++i)
                for (int j = 0; j < 2; ++j) U[i * 2 + j] = Um(i, j);
            apply_one_qubit_unitary(U, wires[0]);
            return;
        }
        if (nw == 2) {
            cplx U[16];
            for (int i = 0; i < 4; ++i)
                for (int j = 0; j < 4; ++j) U[i * 4 + j] = Um(i, j);
            apply_two_qubit_unitary(U, wires[0], wires[1]);
            return;
        }
        std::vector<int> offsets(kdim, 0), local(DIM, 0);
        for (int a = 0; a < kdim; ++a)
            for (int t = 0; t < nw; ++t)
                offsets[a] |= ((a >> (nw - 1 - t)) & 1) << bitpos_for_wire(wires[t]);
        for (int i = 0; i < DIM; ++i)
            for (int wire : wires)
                local[i] = (local[i] << 1) | ((i >> bitpos_for_wire(wire)) & 1);
        std::vector<cplx> tmp(DIM * DIM, cplx(0.0, 0.0));
        std::vector<cplx> result(DIM * DIM, cplx(0.0, 0.0));
        for (int i = 0; i < DIM; ++i)
            for (int j = 0; j < DIM; ++j)
                for (int a = 0; a < kdim; ++a)
                    tmp[i * DIM + j] += Um(local[i], a) * rho_[((i & ~mask) | offsets[a]) * DIM + j];
        for (int i = 0; i < DIM; ++i)
            for (int j = 0; j < DIM; ++j)
                for (int b = 0; b < kdim; ++b)
                    result[i * DIM + j] += tmp[i * DIM + ((j & ~mask) | offsets[b])] * std::conj(Um(local[j], b));
        rho_.swap(result);
    }

    // ---------------- Шум: фазовое и амплитудное затухание -----------------

    void apply_phase_damping(int wire, double gamma) {
        if (wire < 0 || wire >= N) {
            throw std::runtime_error("wire out of range in apply_phase_damping");
        }
        double s = std::sqrt(1.0 - gamma);
        int bp   = bitpos_for_wire(wire);
        int mask = 1 << bp;

        std::vector<cplx> new_rho(DIM * DIM);
        std::fill(new_rho.begin(), new_rho.end(), cplx(0.0, 0.0));

        for (int i = 0; i < DIM; ++i) {
            int bi = (i >> bp) & 1;
            int i0 = i & ~mask;
            int i1 = i0 | mask;

            for (int j = 0; j < DIM; ++j) {
                int bj = (j >> bp) & 1;
                int j0 = j & ~mask;
                int j1 = j0 | mask;

                cplx sum(0.0, 0.0);

                // K0 = diag(1, s)
                {
                    int ia = bi ? i1 : i0;
                    int jb = bj ? j1 : j0;
                    cplx k_i = bi ? s : 1.0;
                    cplx k_j = bj ? s : 1.0;
                    sum += k_i * rho_[ia * DIM + jb] * std::conj(k_j);
                }

                // K1 = diag(0, sqrt(gamma)): только |1><1|
                if (bi && bj) {
                    int ia = i1;
                    int jb = j1;
                    double sg = std::sqrt(gamma);
                    sum += cplx(sg * sg, 0.0) * rho_[ia * DIM + jb];
                }

                new_rho[i * DIM + j] = sum;
            }
        }

        rho_ = new_rho;
    }

    void apply_amplitude_damping(int wire, double gamma) {
        if (wire < 0 || wire >= N) {
            throw std::runtime_error("wire out of range in apply_amplitude_damping");
        }

        // Kraus:
        // K0 = [[1,         0      ],
        //       [0,   sqrt(1-gamma)]]
        //
        // K1 = [[0,       sqrt(gamma)],
        //       [0,             0     ]]
        //
        // В локальном 1-кубитном виде канал даёт:
        // ρ'00 = ρ00 + γ ρ11
        // ρ'01 = √(1-γ) ρ01
        // ρ'10 = √(1-γ) ρ10
        // ρ'11 = (1-γ) ρ11

        double s  = std::sqrt(1.0 - gamma);
        double sq = std::sqrt(gamma);

        int bp   = bitpos_for_wire(wire);
        int mask = 1 << bp;

        std::vector<cplx> new_rho(DIM * DIM);
        std::fill(new_rho.begin(), new_rho.end(), cplx(0.0, 0.0));

        for (int i = 0; i < DIM; ++i) {
            int bi = (i >> bp) & 1;      // бит по данному wire в i
            int i1 = i | mask;           // тот же env, но локальный бит = 1

            for (int j = 0; j < DIM; ++j) {
                int bj = (j >> bp) & 1;
                int j1 = j | mask;       // тот же env, но локальный бит = 1

                // Базовый вклад K0 ρ K0^†
                cplx r = rho_[i * DIM + j];
                cplx k_i0 = (bi == 0) ? cplx(1.0, 0.0) : cplx(s, 0.0);
                cplx k_j0 = (bj == 0) ? cplx(1.0, 0.0) : cplx(s, 0.0);
                cplx val = k_i0 * r * std::conj(k_j0);

                // Дополнительный вклад от K1 ρ K1^†
                // В общем случае он появляется ТОЛЬКО если локальные биты
                // в выходном индексе i,j равны 0 (т.е. мы "прилив" из |1> в |0>):
                if (bi == 0 && bj == 0) {
                    cplx r11 = rho_[i1 * DIM + j1];
                    val += cplx(sq * sq, 0.0) * r11; // γ * ρ_{(bit=1),(bit=1)}
                }

                new_rho[i * DIM + j] = val;
            }
        }

        rho_ = new_rho;
    }

    // ---------------- Общий Kraus на подмножество проводов -----------------

    void apply_kraus(const std::vector<py::array_t<cplx>> &kraus_ops,
                     const std::vector<int> &wires) {
        if (wires.empty()) {
            return;
        }
        int nw   = static_cast<int>(wires.size());
        int kdim = 1 << nw;

        std::vector<std::vector<cplx>> Ks;
        Ks.reserve(kraus_ops.size());
        for (const auto &K_in : kraus_ops) {
            auto r = K_in.request();
            if (r.ndim != 2) {
                throw std::runtime_error("Kraus operator must be 2D");
            }
            int n0 = static_cast<int>(r.shape[0]);
            int n1 = static_cast<int>(r.shape[1]);
            if (n0 != n1 || n0 != kdim) {
                throw std::runtime_error("Kraus operator dimension mismatch");
            }
            auto Km = K_in.unchecked<2>();
            std::vector<cplx> K_flat(kdim * kdim);
            for (int i = 0; i < kdim; ++i) {
                for (int j = 0; j < kdim; ++j) {
                    K_flat[i * kdim + j] = Km(i, j);
                }
            }
            Ks.push_back(std::move(K_flat));
        }

        std::vector<int> bps(nw), masks(nw);
        for (int t = 0; t < nw; ++t) {
            int w = wires[t];
            if (w < 0 || w >= N) {
                throw std::runtime_error("wire out of range in apply_kraus");
            }
            bps[t]   = bitpos_for_wire(w);
            masks[t] = 1 << bps[t];
        }

        std::vector<cplx> new_rho(DIM * DIM);
        std::fill(new_rho.begin(), new_rho.end(), cplx(0.0, 0.0));

        for (int i = 0; i < DIM; ++i) {
            int li   = 0;
            int env_i = i;
            for (int t = 0; t < nw; ++t) {
                int bp  = bps[t];
                int bit = (i >> bp) & 1;
                env_i  &= ~masks[t];
                li = (li << 1) | bit;
            }

            for (int j = 0; j < DIM; ++j) {
                int lj    = 0;
                int env_j = j;
                for (int t = 0; t < nw; ++t) {
                    int bp  = bps[t];
                    int bit = (j >> bp) & 1;
                    env_j  &= ~masks[t];
                    lj = (lj << 1) | bit;
                }

                cplx sum(0.0, 0.0);

                for (const auto &K : Ks) {
                    for (int a = 0; a < kdim; ++a) {
                        int ia = env_i;
                        for (int t = 0; t < nw; ++t) {
                            int bit = (a >> (nw - 1 - t)) & 1;
                            ia |= (bit << bps[t]);
                        }
                        cplx k_left = K[li * kdim + a];

                        for (int b = 0; b < kdim; ++b) {
                            int jb = env_j;
                            for (int t = 0; t < nw; ++t) {
                                int bit = (b >> (nw - 1 - t)) & 1;
                                jb |= (bit << bps[t]);
                            }
                            cplx k_right = std::conj(K[lj * kdim + b]);
                            sum += k_left * rho_[ia * DIM + jb] * k_right;
                        }
                    }
                }

                new_rho[i * DIM + j] = sum;
            }
        }

        rho_ = new_rho;
    }

    // ---------------- Доступ для pickle -----------------

    const std::vector<cplx> &rho() const { return rho_; }
    const std::mt19937 &rng() const { return rng_; }

    void set_rho(const std::vector<cplx> &r) { rho_ = r; }
    void set_rng(const std::mt19937 &r) { rng_ = r; }


private:
    std::vector<cplx> rho_;
    std::mt19937 rng_;
};

struct EnginePickleAccess {
    static py::tuple getstate(const Engine &e) {
        py::array_t<cplx> rho_arr({DIM, DIM});
        auto buf = rho_arr.mutable_unchecked<2>();
        const auto &rho = e.rho();
        for (int i = 0; i < DIM; ++i) {
            for (int j = 0; j < DIM; ++j) {
                buf(i, j) = rho[i * DIM + j];
            }
        }

        std::vector<std::uint32_t> rng_state(624);
        std::mt19937 rng_copy = e.rng();
        for (size_t i = 0; i < rng_state.size(); ++i) {
            rng_state[i] = rng_copy();
        }

        return py::make_tuple(rho_arr, rng_state);
    }

    static Engine setstate(py::tuple t) {
        if (t.size() != 2) {
            throw std::runtime_error("Invalid state for Engine");
        }
        Engine e;

        py::array_t<cplx> rho_arr = t[0].cast<py::array_t<cplx>>();
        auto buf = rho_arr.unchecked<2>();
        if (buf.shape(0) != DIM || buf.shape(1) != DIM) {
            throw std::runtime_error("Invalid rho shape in pickle");
        }

        std::vector<cplx> rho(DIM * DIM);
        for (int i = 0; i < DIM; ++i) {
            for (int j = 0; j < DIM; ++j) {
                rho[i * DIM + j] = buf(i, j);
            }
        }
        e.set_rho(rho);

        std::vector<std::uint32_t> rng_state = t[1].cast<std::vector<std::uint32_t>>();
        std::seed_seq seq(rng_state.begin(), rng_state.end());
        std::mt19937 rng(seq);
        e.set_rng(rng);

        return e;
    }
};

PYBIND11_MODULE(MODULE_NAME, m) {
    py::class_<Engine>(m, "Engine", py::module_local())
        .def(py::init<>())
        .def("reset", &Engine::reset)
        .def("set_basis_state", &Engine::set_basis_state)
        .def("get_state", &Engine::get_state)
        .def("get_probs", &Engine::get_probs)
        .def("get_probs_marginal", &Engine::get_probs_marginal)
        .def("generate_samples", &Engine::generate_samples)
        .def("count_states", &Engine::count_states)
        .def("apply_rx", &Engine::apply_rx)
        .def("apply_ry", &Engine::apply_ry)
        .def("apply_rz", &Engine::apply_rz)
        .def("apply_phase_shift", &Engine::apply_phase_shift)
        .def("apply_berkeley", &Engine::apply_berkeley)
        .def("apply_cnot", &Engine::apply_cnot)
        .def("apply_crx", &Engine::apply_crx)
        .def("apply_cphase", &Engine::apply_cphase)
        .def("apply_unitary", &Engine::apply_unitary)
        .def("apply_phase_damping", &Engine::apply_phase_damping)
        .def("apply_amplitude_damping", &Engine::apply_amplitude_damping)
        .def("apply_kraus", &Engine::apply_kraus)
        .def(py::pickle(
            [](const Engine &e) {
                return EnginePickleAccess::getstate(e);
            },
            [](py::tuple t) {
                return EnginePickleAccess::setstate(t);
            }
        ));
}
