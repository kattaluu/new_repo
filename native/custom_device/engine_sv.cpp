// engine_sv.cpp – statevector-движок с корректным сэмплированием

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/complex.h>
#include <pybind11/stl.h>

#include <complex>
#include <cstdint>
#include <vector>
#include <cmath>
#include <random>
#include <algorithm>
#include <stdexcept>

#ifndef NUM_QUBITS
#define NUM_QUBITS 4
#endif

#ifndef MODULE_NAME
#define MODULE_NAME engine4q_sv
#endif

namespace py = pybind11;
using cplx = std::complex<double>;

static constexpr int N   = NUM_QUBITS;
static constexpr int DIM = 1 << N;

// PennyLane-конвенция: wire=0 — ЛЕВЫЙ кубит в |q0 q1 ... q_{N-1}>.
// Внутренний индекс состояния i — двоичное число с младшим битом справа.
// => позиция бита для wire: bitpos = N-1-wire.
inline int bitpos_for_wire(int wire) {
    return N - 1 - wire;
}

class EngineSV {
public:
    EngineSV()
    : psi_(DIM, cplx(0.0, 0.0)), rng_(std::random_device{}())
    {
        reset();
    }

    // ---------------- Базовые операции -----------------

    void reset() {
        std::fill(psi_.begin(), psi_.end(), cplx(0.0, 0.0));
        psi_[0] = cplx(1.0, 0.0);  // |0...0>
    }

    void set_basis_state(int index) {
        if (index < 0 || index >= DIM) {
            throw std::runtime_error("basis index out of range");
        }
        std::fill(psi_.begin(), psi_.end(), cplx(0.0, 0.0));
        psi_[index] = cplx(1.0, 0.0);
    }

    py::array_t<cplx> get_state() const {
        py::array_t<cplx> out(DIM);
        auto buf = out.mutable_unchecked<1>();
        for (int i = 0; i < DIM; ++i) {
            buf(i) = psi_[i];
        }
        return out;
    }

    // p(i) = |psi_i|^2
    py::array_t<double> get_probs() const {
        py::array_t<double> out(DIM);
        auto buf = out.mutable_unchecked<1>();
        double norm = 0.0;
        for (int i = 0; i < DIM; ++i) {
            double p = std::norm(psi_[i]);
            buf(i) = p;
            norm += p;
        }
        if (norm <= 0.0) {
            // деградировавшее состояние – вернём |0...0>
            for (int i = 0; i < DIM; ++i) buf(i) = 0.0;
            buf(0) = 1.0;
            return out;
        }
        for (int i = 0; i < DIM; ++i) {
            buf(i) /= norm;
        }
        return out;
    }

    // Маргинальные вероятности по подмножеству wires
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

        for (int i = 0; i < DIM; ++i) {
            double p = std::norm(psi_[i]);
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
            double p = std::norm(psi_[i]);
            probs[i] = p;
            norm += p;
        }
        if (norm <= 0.0) {
            std::fill(probs.begin(), probs.end(), 0.0);
            probs[0] = 1.0;
            norm = 1.0;
        } else {
            for (double &p : probs) p /= norm;
        }

        std::discrete_distribution<int> dist(probs.begin(), probs.end());

        // samples[s, j], где j=0 соответствует wire=0 (левый кубит)
        py::array_t<std::uint8_t> out({shots, N});
        auto buf = out.mutable_unchecked<2>();

        for (int s = 0; s < shots; ++s) {
            int idx = dist(rng_);
            for (int j = 0; j < N; ++j) {
                int bp  = N - 1 - j;              // wire j → bitpos N-1-j
                std::uint8_t bit = (idx >> bp) & 1;
                buf(s, j) = bit;
            }
        }

        return out;
    }

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
                std::uint8_t bit = sm(s, w);  // w – индекс провода (тот же, что в девайсе)
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

        std::vector<cplx> new_psi(DIM, cplx(0.0, 0.0));

        // Обходим пары (i0, i1) с фиксированным env, меняя только локальный бит
        for (int base = 0; base < DIM; ++base) {
            if ((base & mask) != 0) continue;  // берём только те, где локальный бит=0
            int i0 = base;
            int i1 = base | mask;

            cplx v0 = psi_[i0];
            cplx v1 = psi_[i1];

            new_psi[i0] = U[0] * v0 + U[1] * v1;
            new_psi[i1] = U[2] * v0 + U[3] * v1;
        }

        psi_.swap(new_psi);
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

        std::vector<cplx> new_psi(DIM, cplx(0.0, 0.0));

        // Обходим блоки по 4 состояния для (wire0, wire1)
        for (int base = 0; base < DIM; ++base) {
            if ((base & mask0) != 0 || (base & mask1) != 0) continue;

            int i00 = base;
            int i10 = base | mask0;
            int i01 = base | mask1;
            int i11 = base | mask0 | mask1;

            cplx v[4] = { psi_[i00], psi_[i01], psi_[i10], psi_[i11] };
            cplx res[4] = { cplx(0.0, 0.0), cplx(0.0, 0.0), cplx(0.0, 0.0), cplx(0.0, 0.0) };

            for (int r = 0; r < 4; ++r) {
                for (int c = 0; c < 4; ++c) {
                    res[r] += U[r * 4 + c] * v[c];
                }
            }

            new_psi[i00] = res[0];
            new_psi[i01] = res[1];
            new_psi[i10] = res[2];
            new_psi[i11] = res[3];
        }

        psi_.swap(new_psi);
    }

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
        // Матрица CNOT в базисе |00>,|10>,|01>,|11> (control, target)
        cplx U[16] = {};
        U[0 * 4 + 0] = cplx(1.0, 0.0); // |00> -> |00|
        U[1 * 4 + 1] = cplx(1.0, 0.0); // |10> -> |10|
        U[2 * 4 + 3] = cplx(1.0, 0.0); // |01> -> |11|
        U[3 * 4 + 2] = cplx(1.0, 0.0); // |11> -> |01|
        apply_two_qubit_unitary(U, control, target);
    }

    void apply_crx(double theta, int control, int target) {
        double c = std::cos(theta / 2.0);
        double s = std::sin(theta / 2.0);
        cplx minus_i(0.0, -1.0);

        cplx U[16] = {};
        // |00>, |10> остаются как есть
        U[0 * 4 + 0] = cplx(1.0, 0.0); // |00>
        U[1 * 4 + 1] = cplx(1.0, 0.0); // |10>

        // подпространство target при control=1: |01>,|11|
        // индекс 2 -> |01>, 3 -> |11>, если порядок (control, target)
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
        U[1 * 4 + 1] = cplx(1.0, 0.0); // |10>
        U[2 * 4 + 2] = cplx(1.0, 0.0); // |01>
        U[3 * 4 + 3] = phase;          // |11>
        apply_two_qubit_unitary(U, control, target);
    }

    // Общий unitar: либо 2x2 на один кубит, либо DIMxDIM (редко)
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
        std::vector<cplx> result(DIM, cplx(0.0, 0.0));
        for (int i = 0; i < DIM; ++i)
            for (int a = 0; a < kdim; ++a)
                result[i] += Um(local[i], a) * psi_[(i & ~mask) | offsets[a]];
        psi_.swap(result);
    }

    // -------- pickle-доступ --------

    const std::vector<cplx> &state() const { return psi_; }
    const std::mt19937 &rng() const { return rng_; }

    void set_state(const std::vector<cplx> &st) {
        if (static_cast<int>(st.size()) != DIM) {
            throw std::runtime_error("set_state: wrong size");
        }
        psi_ = st;
    }

    void set_rng(const std::mt19937 &r) { rng_ = r; }

private:
    std::vector<cplx> psi_;
    std::mt19937 rng_;
};

struct EngineSVPickleAccess {
    static py::tuple getstate(const EngineSV &e) {
        py::array_t<cplx> psi_arr(DIM);
        auto buf = psi_arr.mutable_unchecked<1>();
        const auto &st = e.state();
        for (int i = 0; i < DIM; ++i) {
            buf(i) = st[i];
        }

        std::vector<std::uint32_t> rng_state(624);
        std::mt19937 rng_copy = e.rng();
        for (size_t i = 0; i < rng_state.size(); ++i) {
            rng_state[i] = rng_copy();
        }

        return py::make_tuple(psi_arr, rng_state);
    }

    static EngineSV setstate(py::tuple t) {
        if (t.size() != 2) {
            throw std::runtime_error("Invalid state for EngineSV");
        }
        EngineSV e;

        py::array_t<cplx> psi_arr = t[0].cast<py::array_t<cplx>>();
        auto buf = psi_arr.unchecked<1>();
        if (buf.shape(0) != DIM) {
            throw std::runtime_error("Invalid psi shape in pickle");
        }

        std::vector<cplx> psi(DIM);
        for (int i = 0; i < DIM; ++i) {
            psi[i] = buf(i);
        }
        e.set_state(psi);

        std::vector<std::uint32_t> rng_state = t[1].cast<std::vector<std::uint32_t>>();
        std::seed_seq seq(rng_state.begin(), rng_state.end());
        std::mt19937 rng(seq);
        e.set_rng(rng);

        return e;
    }
};

PYBIND11_MODULE(MODULE_NAME, m) {
    py::class_<EngineSV>(m, "Engine", py::module_local())
        .def(py::init<>())
        .def("reset", &EngineSV::reset)
        .def("set_basis_state", &EngineSV::set_basis_state)
        .def("get_state", &EngineSV::get_state)
        .def("get_probs", &EngineSV::get_probs)
        .def("get_probs_marginal", &EngineSV::get_probs_marginal)
        .def("generate_samples", &EngineSV::generate_samples)
        .def("count_states", &EngineSV::count_states)
        .def("apply_rx", &EngineSV::apply_rx)
        .def("apply_ry", &EngineSV::apply_ry)
        .def("apply_rz", &EngineSV::apply_rz)
        .def("apply_berkeley", &EngineSV::apply_berkeley)
        .def("apply_cnot", &EngineSV::apply_cnot)
        .def("apply_crx", &EngineSV::apply_crx)
        .def("apply_cphase", &EngineSV::apply_cphase)
        .def("apply_unitary", &EngineSV::apply_unitary)
        .def(py::pickle(
            [](const EngineSV &e) {
                return EngineSVPickleAccess::getstate(e);
            },
            [](py::tuple t) {
                return EngineSVPickleAccess::setstate(t);
            }
        ));
}
