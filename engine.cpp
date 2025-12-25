#pragma once

#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <algorithm>
#include <vector>
#include <memory>
#include <type_traits>
#include <atomic>
#include <mutex>
#include <condition_variable>
#include <queue>
#include <functional>
#include <stdexcept>
#include <iostream>
#include <iomanip>
#include <chrono>
#include <bitset>
#include <complex>
#include <random>
#include <numeric>
#include <variant>
#include <optional>
#include <string_view>
#include <array>
#include <limits>
#include <tuple>
#include <unordered_map>
#include <set>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <shared_mutex>

#if defined(__x86_64__) || defined(_M_X64)
    #define ARCH_X86
#endif

#if defined(__aarch64__) || defined(_M_ARM64)
    #define ARCH_ARM
#endif

#ifdef ARCH_X86
#include <xmmintrin.h>
#include <emmintrin.h>
#include <immintrin.h>
#endif

#ifdef ARCH_ARM
#include <arm_neon.h>
#endif

namespace quantum {

class SystemInvariants {
public:
    static constexpr size_t MAX_SUPPORTED_QUBITS = 30;
    static constexpr size_t MIN_SUPPORTED_QUBITS = 1;
    static constexpr size_t MAX_CIRCUIT_DEPTH = 10000;
    static constexpr double MAX_GRADIENT_MAGNITUDE = 100.0;
    static constexpr double MIN_NUMERICAL_VALUE = 1e-15;
    static constexpr double MAX_NUMERICAL_VALUE = 1e15;
    
    static bool validate_qubit_count(size_t qubits) {
        return qubits >= MIN_SUPPORTED_QUBITS && qubits <= MAX_SUPPORTED_QUBITS;
    }
    
    static bool validate_gradient_value(double grad) {
        return std::isfinite(grad) && std::abs(grad) <= MAX_GRADIENT_MAGNITUDE;
    }
    
    static bool validate_numerical_value(double val) {
        return std::isfinite(val) && std::abs(val) <= MAX_NUMERICAL_VALUE;
    }
};

class ThreadSafeRNG {
    static std::atomic<uint64_t> global_seed_counter_;
    thread_local static std::mt19937_64 thread_engine_;
    thread_local static std::optional<uint64_t> thread_seed_;
    thread_local static bool thread_deterministic_;
    
public:
    ThreadSafeRNG() {
        if (!thread_seed_.has_value()) {
            thread_engine_.seed(global_seed_counter_.fetch_add(1, std::memory_order_relaxed));
            thread_deterministic_ = false;
        }
    }
    
    explicit ThreadSafeRNG(uint64_t seed) {
        thread_engine_.seed(seed);
        thread_seed_ = seed;
        thread_deterministic_ = true;
    }
    
    double uniform() {
        std::uniform_real_distribution<double> dist(0.0, 1.0);
        return dist(thread_engine_);
    }
    
    size_t discrete(const std::vector<double>& probabilities) {
        std::discrete_distribution<size_t> dist(probabilities.begin(), probabilities.end());
        return dist(thread_engine_);
    }
    
    bool bernoulli(double p) {
        std::bernoulli_distribution dist(p);
        return dist(thread_engine_);
    }
    
    bool is_deterministic() const { return thread_deterministic_; }
    std::optional<uint64_t> get_seed() const { return thread_seed_; }
    
    std::string get_state() const {
        std::ostringstream oss;
        oss << thread_engine_;
        return oss.str();
    }
    
    void set_state(const std::string& state) {
        std::istringstream iss(state);
        iss >> thread_engine_;
        thread_deterministic_ = true;
    }
    
    static void reset_global_counter() {
        global_seed_counter_.store(0, std::memory_order_relaxed);
    }
};

std::atomic<uint64_t> ThreadSafeRNG::global_seed_counter_{0};
thread_local std::mt19937_64 ThreadSafeRNG::thread_engine_;
thread_local std::optional<uint64_t> ThreadSafeRNG::thread_seed_;
thread_local bool ThreadSafeRNG::thread_deterministic_;

class QuantumState {
    struct StateData {
        std::unique_ptr<std::complex<double>[]> amplitudes;
        size_t num_qubits;
        size_t dimension;
        bool measured;
        
        StateData(size_t n_qubits) 
            : num_qubits(n_qubits), 
              dimension(1ULL << n_qubits),
              measured(false) {
            if (!SystemInvariants::validate_qubit_count(n_qubits)) {
                throw std::invalid_argument("Invalid number of qubits");
            }
            
            amplitudes = std::make_unique<std::complex<double>[]>(dimension);
            amplitudes[0] = 1.0;
            std::fill(amplitudes.get() + 1, amplitudes.get() + dimension, 0.0);
        }
    };
    
    std::shared_ptr<StateData> data_;
    std::shared_ptr<ThreadSafeRNG> rng_;
    
    void validate_qubit_index(size_t qubit) const {
        if (qubit >= data_->num_qubits) {
            throw std::out_of_range("Qubit index out of range");
        }
    }
    
    void validate_coherent(const char* operation) const {
        if (data_->measured) {
            throw std::logic_error(std::string(operation) + 
                                  " requires coherent quantum state");
        }
    }
    
    void normalize() {
        double norm = 0.0;
        for (size_t i = 0; i < data_->dimension; ++i) {
            norm += std::norm(data_->amplitudes[i]);
        }
        
        if (norm > SystemInvariants::MIN_NUMERICAL_VALUE) {
            double inv_norm = 1.0 / std::sqrt(norm);
            for (size_t i = 0; i < data_->dimension; ++i) {
                data_->amplitudes[i] *= inv_norm;
            }
        }
    }
    
public:
    QuantumState(size_t num_qubits, std::shared_ptr<ThreadSafeRNG> rng = nullptr)
        : data_(std::make_shared<StateData>(num_qubits)), rng_(rng) {
        if (!rng_) {
            rng_ = std::make_shared<ThreadSafeRNG>();
        }
    }
    
    size_t num_qubits() const { return data_->num_qubits; }
    bool is_measured() const { return data_->measured; }
    bool is_coherent() const { return !data_->measured; }
    
    void apply_hadamard(size_t qubit) {
        validate_coherent("Hadamard gate");
        validate_qubit_index(qubit);
        
        const size_t stride = 1ULL << qubit;
        const double factor = M_SQRT1_2;
        
        for (size_t i = 0; i < data_->dimension; i += 2 * stride) {
            for (size_t j = 0; j < stride; ++j) {
                size_t idx0 = i + j;
                size_t idx1 = i + j + stride;
                
                std::complex<double> a = data_->amplitudes[idx0];
                std::complex<double> b = data_->amplitudes[idx1];
                
                data_->amplitudes[idx0] = (a + b) * factor;
                data_->amplitudes[idx1] = (a - b) * factor;
            }
        }
    }
    
    void apply_cnot(size_t control, size_t target) {
        validate_coherent("CNOT gate");
        validate_qubit_index(control);
        validate_qubit_index(target);
        
        if (control == target) return;
        
        const size_t control_mask = 1ULL << control;
        const size_t target_mask = 1ULL << target;
        
        for (size_t i = 0; i < data_->dimension; ++i) {
            if ((i & control_mask) && !(i & target_mask)) {
                size_t j = i ^ target_mask;
                if (i < j) {
                    std::swap(data_->amplitudes[i], data_->amplitudes[j]);
                }
            }
        }
    }
    
    void apply_rz(double angle, size_t qubit) {
        validate_coherent("RZ gate");
        validate_qubit_index(qubit);
        
        const size_t stride = 1ULL << qubit;
        std::complex<double> phase(std::cos(angle), std::sin(angle));
        
        for (size_t i = 0; i < data_->dimension; i += 2 * stride) {
            for (size_t j = 0; j < stride; ++j) {
                data_->amplitudes[i + j + stride] *= phase;
            }
        }
    }
    
    void apply_ry(double angle, size_t qubit) {
        validate_coherent("RY gate");
        validate_qubit_index(qubit);
        
        const size_t stride = 1ULL << qubit;
        double c = std::cos(angle * 0.5);
        double s = std::sin(angle * 0.5);
        
        for (size_t i = 0; i < data_->dimension; i += 2 * stride) {
            for (size_t j = 0; j < stride; ++j) {
                size_t idx0 = i + j;
                size_t idx1 = i + j + stride;
                
                std::complex<double> a = data_->amplitudes[idx0];
                std::complex<double> b = data_->amplitudes[idx1];
                
                data_->amplitudes[idx0] = c * a - s * b;
                data_->amplitudes[idx1] = s * a + c * b;
            }
        }
    }
    
    void apply_rx(double angle, size_t qubit) {
        validate_coherent("RX gate");
        validate_qubit_index(qubit);
        
        const size_t stride = 1ULL << qubit;
        double c = std::cos(angle * 0.5);
        double s = std::sin(angle * 0.5);
        
        for (size_t i = 0; i < data_->dimension; i += 2 * stride) {
            for (size_t j = 0; j < stride; ++j) {
                size_t idx0 = i + j;
                size_t idx1 = i + j + stride;
                
                std::complex<double> a = data_->amplitudes[idx0];
                std::complex<double> b = data_->amplitudes[idx1];
                
                data_->amplitudes[idx0] = c * a - std::complex<double>(0.0, s) * b;
                data_->amplitudes[idx1] = std::complex<double>(0.0, -s) * a + c * b;
            }
        }
    }
    
    void apply_pauli_x(size_t qubit) {
        validate_coherent("Pauli X gate");
        validate_qubit_index(qubit);
        
        const size_t stride = 1ULL << qubit;
        
        for (size_t i = 0; i < data_->dimension; i += 2 * stride) {
            for (size_t j = 0; j < stride; ++j) {
                std::swap(data_->amplitudes[i + j], data_->amplitudes[i + j + stride]);
            }
        }
    }
    
    void apply_pauli_y(size_t qubit) {
        validate_coherent("Pauli Y gate");
        validate_qubit_index(qubit);
        
        const size_t stride = 1ULL << qubit;
        
        for (size_t i = 0; i < data_->dimension; i += 2 * stride) {
            for (size_t j = 0; j < stride; ++j) {
                size_t idx0 = i + j;
                size_t idx1 = i + j + stride;
                
                std::complex<double> a = data_->amplitudes[idx0];
                std::complex<double> b = data_->amplitudes[idx1];
                
                data_->amplitudes[idx0] = std::complex<double>(0.0, -1.0) * b;
                data_->amplitudes[idx1] = std::complex<double>(0.0, 1.0) * a;
            }
        }
    }
    
    void apply_pauli_z(size_t qubit) {
        validate_coherent("Pauli Z gate");
        validate_qubit_index(qubit);
        
        const size_t stride = 1ULL << qubit;
        
        for (size_t i = 0; i < data_->dimension; i += 2 * stride) {
            for (size_t j = 0; j < stride; ++j) {
                data_->amplitudes[i + j + stride] *= -1.0;
            }
        }
    }
    
    void apply_cz(size_t control, size_t target) {
        validate_coherent("CZ gate");
        validate_qubit_index(control);
        validate_qubit_index(target);
        
        if (control == target) return;
        
        const size_t control_mask = 1ULL << control;
        const size_t target_mask = 1ULL << target;
        
        for (size_t i = 0; i < data_->dimension; ++i) {
            if ((i & control_mask) && (i & target_mask)) {
                data_->amplitudes[i] *= -1.0;
            }
        }
    }
    
    size_t measure_full() {
        validate_coherent("Full measurement");
        
        std::vector<double> probabilities(data_->dimension);
        for (size_t i = 0; i < data_->dimension; ++i) {
            probabilities[i] = std::norm(data_->amplitudes[i]);
        }
        
        size_t result = rng_->discrete(probabilities);
        
        std::fill(data_->amplitudes.get(), data_->amplitudes.get() + data_->dimension, 0.0);
        data_->amplitudes[result] = 1.0;
        
        data_->measured = true;
        return result;
    }
    
    std::vector<bool> measure_subset(const std::vector<size_t>& qubits) {
        validate_coherent("Partial measurement");
        
        std::vector<bool> results;
        results.reserve(qubits.size());
        
        for (size_t qubit : qubits) {
            validate_qubit_index(qubit);
            size_t mask = 1ULL << qubit;
            double prob_one = 0.0;
            
            for (size_t i = 0; i < data_->dimension; ++i) {
                if (i & mask) {
                    prob_one += std::norm(data_->amplitudes[i]);
                }
            }
            
            bool measured_one = rng_->bernoulli(prob_one);
            results.push_back(measured_one);
            
            double scale = measured_one ? (prob_one > 0.0 ? 1.0 / std::sqrt(prob_one) : 0.0) :
                                        ((1.0 - prob_one) > 0.0 ? 1.0 / std::sqrt(1.0 - prob_one) : 0.0);
            
            for (size_t i = 0; i < data_->dimension; ++i) {
                if ((i & mask) != (measured_one ? mask : 0)) {
                    data_->amplitudes[i] = 0.0;
                } else if (scale != 0.0) {
                    data_->amplitudes[i] *= scale;
                }
            }
        }
        
        data_->measured = true;
        return results;
    }
    
    std::vector<double> get_probabilities() const {
        std::vector<double> probs(data_->dimension);
        
        if (data_->measured) {
            for (size_t i = 0; i < data_->dimension; ++i) {
                probs[i] = (std::abs(data_->amplitudes[i]) > 0.5) ? 1.0 : 0.0;
            }
        } else {
            for (size_t i = 0; i < data_->dimension; ++i) {
                probs[i] = std::norm(data_->amplitudes[i]);
            }
        }
        
        return probs;
    }
    
    double expectation_pauli_z(size_t qubit) const {
        validate_qubit_index(qubit);
        
        double expectation = 0.0;
        size_t mask = 1ULL << qubit;
        
        if (data_->measured) {
            for (size_t i = 0; i < data_->dimension; ++i) {
                if (std::abs(data_->amplitudes[i]) > 0.5) {
                    expectation += ((i & mask) ? -1.0 : 1.0);
                }
            }
        } else {
            for (size_t i = 0; i < data_->dimension; ++i) {
                double prob = std::norm(data_->amplitudes[i]);
                expectation += prob * ((i & mask) ? -1.0 : 1.0);
            }
        }
        
        return expectation;
    }
    
    std::shared_ptr<ThreadSafeRNG> get_rng() const { return rng_; }
};

class ParameterizedCircuit {
    struct Gate {
        enum class Type { RZ, RY, RX, H, X, Y, Z, CNOT, CZ };
        Type type;
        size_t target;
        size_t control;
        size_t param_index;
        double fixed_param;
    };
    
    std::vector<Gate> gates_;
    size_t num_qubits_;
    size_t num_params_;
    bool compiled_;
    
    void validate_gate(const Gate& gate) const {
        if (gate.type == Gate::Type::CNOT || gate.type == Gate::Type::CZ) {
            if (gate.control >= num_qubits_ || gate.target >= num_qubits_) {
                throw std::out_of_range("Qubit index out of range");
            }
            if (gate.control == gate.target) {
                throw std::invalid_argument("Control and target qubits must be different");
            }
        } else {
            if (gate.target >= num_qubits_) {
                throw std::out_of_range("Qubit index out of range");
            }
        }
        
        if (gate.param_index >= num_params_) {
            throw std::out_of_range("Parameter index out of range");
        }
    }
    
public:
    ParameterizedCircuit(size_t num_qubits, size_t num_params)
        : num_qubits_(num_qubits), num_params_(num_params), compiled_(false) {
        if (!SystemInvariants::validate_qubit_count(num_qubits)) {
            throw std::invalid_argument("Invalid number of qubits");
        }
    }
    
    void add_rz_gate(size_t qubit, size_t param_idx) {
        if (compiled_) throw std::logic_error("Circuit already compiled");
        gates_.push_back({Gate::Type::RZ, qubit, 0, param_idx, 0.0});
    }
    
    void add_ry_gate(size_t qubit, size_t param_idx) {
        if (compiled_) throw std::logic_error("Circuit already compiled");
        gates_.push_back({Gate::Type::RY, qubit, 0, param_idx, 0.0});
    }
    
    void add_rx_gate(size_t qubit, size_t param_idx) {
        if (compiled_) throw std::logic_error("Circuit already compiled");
        gates_.push_back({Gate::Type::RX, qubit, 0, param_idx, 0.0});
    }
    
    void add_hadamard_gate(size_t qubit) {
        if (compiled_) throw std::logic_error("Circuit already compiled");
        gates_.push_back({Gate::Type::H, qubit, 0, 0, 0.0});
    }
    
    void add_pauli_x_gate(size_t qubit) {
        if (compiled_) throw std::logic_error("Circuit already compiled");
        gates_.push_back({Gate::Type::X, qubit, 0, 0, 0.0});
    }
    
    void add_pauli_y_gate(size_t qubit) {
        if (compiled_) throw std::logic_error("Circuit already compiled");
        gates_.push_back({Gate::Type::Y, qubit, 0, 0, 0.0});
    }
    
    void add_pauli_z_gate(size_t qubit) {
        if (compiled_) throw std::logic_error("Circuit already compiled");
        gates_.push_back({Gate::Type::Z, qubit, 0, 0, 0.0});
    }
    
    void add_cnot_gate(size_t control, size_t target) {
        if (compiled_) throw std::logic_error("Circuit already compiled");
        gates_.push_back({Gate::Type::CNOT, target, control, 0, 0.0});
    }
    
    void add_cz_gate(size_t control, size_t target) {
        if (compiled_) throw std::logic_error("Circuit already compiled");
        gates_.push_back({Gate::Type::CZ, target, control, 0, 0.0});
    }
    
    void compile() {
        for (const auto& gate : gates_) {
            validate_gate(gate);
        }
        compiled_ = true;
    }
    
    void execute(QuantumState& state, const std::vector<double>& parameters) const {
        if (!compiled_) {
            throw std::logic_error("Circuit must be compiled before execution");
        }
        
        if (parameters.size() != num_params_) {
            throw std::invalid_argument("Parameter count mismatch");
        }
        
        for (const auto& gate : gates_) {
            switch (gate.type) {
                case Gate::Type::RZ:
                    state.apply_rz(parameters[gate.param_index], gate.target);
                    break;
                case Gate::Type::RY:
                    state.apply_ry(parameters[gate.param_index], gate.target);
                    break;
                case Gate::Type::RX:
                    state.apply_rx(parameters[gate.param_index], gate.target);
                    break;
                case Gate::Type::H:
                    state.apply_hadamard(gate.target);
                    break;
                case Gate::Type::X:
                    state.apply_pauli_x(gate.target);
                    break;
                case Gate::Type::Y:
                    state.apply_pauli_y(gate.target);
                    break;
                case Gate::Type::Z:
                    state.apply_pauli_z(gate.target);
                    break;
                case Gate::Type::CNOT:
                    state.apply_cnot(gate.control, gate.target);
                    break;
                case Gate::Type::CZ:
                    state.apply_cz(gate.control, gate.target);
                    break;
            }
        }
    }
    
    size_t num_qubits() const { return num_qubits_; }
    size_t num_parameters() const { return num_params_; }
    size_t num_gates() const { return gates_.size(); }
    bool is_compiled() const { return compiled_; }
};

class Observable {
public:
    virtual ~Observable() = default;
    virtual double expectation(const QuantumState& state) const = 0;
    virtual std::string name() const = 0;
    virtual std::unique_ptr<Observable> clone() const = 0;
};

class PauliZObservable : public Observable {
    size_t qubit_;
    
public:
    explicit PauliZObservable(size_t qubit) : qubit_(qubit) {}
    
    double expectation(const QuantumState& state) const override {
        return state.expectation_pauli_z(qubit_);
    }
    
    std::string name() const override {
        return "PauliZ(" + std::to_string(qubit_) + ")";
    }
    
    std::unique_ptr<Observable> clone() const override {
        return std::make_unique<PauliZObservable>(*this);
    }
};

class QuantumModel {
public:
    struct ModelState {
        std::vector<double> parameters;
        std::string rng_state;
        
        bool operator==(const ModelState& other) const {
            if (parameters.size() != other.parameters.size()) return false;
            for (size_t i = 0; i < parameters.size(); ++i) {
                if (std::abs(parameters[i] - other.parameters[i]) > 1e-10) return false;
            }
            return rng_state == other.rng_state;
        }
    };
    
private:
    std::shared_ptr<ParameterizedCircuit> circuit_;
    std::vector<double> parameters_;
    std::unique_ptr<Observable> observable_;
    std::shared_ptr<ThreadSafeRNG> rng_;
    
    static double smooth_clip(double grad, double max_mag) {
        const double scale = grad / max_mag;
        return max_mag * std::tanh(scale);
    }
    
    double compute_expectation(const std::vector<double>& eval_params) const {
        QuantumState state(circuit_->num_qubits(), rng_);
        circuit_->execute(state, eval_params);
        return observable_->expectation(state);
    }
    
public:
    QuantumModel(std::shared_ptr<ParameterizedCircuit> circuit,
                const std::vector<double>& initial_params,
                std::unique_ptr<Observable> observable,
                std::shared_ptr<ThreadSafeRNG> rng = nullptr)
        : circuit_(circuit), parameters_(initial_params), 
          observable_(std::move(observable)), rng_(rng) {
        
        if (!circuit_->is_compiled()) {
            circuit_->compile();
        }
        
        if (parameters_.size() != circuit_->num_parameters()) {
            throw std::invalid_argument("Parameter count mismatch");
        }
        
        if (!rng_) {
            rng_ = std::make_shared<ThreadSafeRNG>();
        }
    }
    
    double expectation() const {
        return compute_expectation(parameters_);
    }
    
    double loss(double target = 1.0) const {
        return std::abs(expectation() - target);
    }
    
    std::vector<double> compute_gradients(double epsilon = M_PI_2) const {
        std::vector<double> gradients(parameters_.size());
        
        for (size_t i = 0; i < parameters_.size(); ++i) {
            std::vector<double> params_plus = parameters_;
            std::vector<double> params_minus = parameters_;
            
            params_plus[i] += epsilon;
            params_minus[i] -= epsilon;
            
            double exp_plus = compute_expectation(params_plus);
            double exp_minus = compute_expectation(params_minus);
            
            gradients[i] = 0.5 * (exp_plus - exp_minus);
            
            if (!SystemInvariants::validate_gradient_value(gradients[i])) {
                gradients[i] = smooth_clip(gradients[i], SystemInvariants::MAX_GRADIENT_MAGNITUDE);
            }
        }
        
        return gradients;
    }
    
    void update_parameters(const std::vector<double>& gradients, double learning_rate) {
        if (gradients.size() != parameters_.size()) {
            throw std::invalid_argument("Gradient size mismatch");
        }
        
        double grad_norm = 0.0;
        for (double g : gradients) {
            grad_norm += g * g;
        }
        grad_norm = std::sqrt(grad_norm);
        
        const double max_norm = SystemInvariants::MAX_GRADIENT_MAGNITUDE;
        double scale = (grad_norm > max_norm && grad_norm > 0.0) ? max_norm / grad_norm : 1.0;
        
        for (size_t i = 0; i < parameters_.size(); ++i) {
            double grad = gradients[i] * scale;
            
            if (!SystemInvariants::validate_gradient_value(grad)) {
                grad = smooth_clip(grad, SystemInvariants::MAX_GRADIENT_MAGNITUDE);
            }
            
            parameters_[i] -= learning_rate * grad;
            
            if (!SystemInvariants::validate_numerical_value(parameters_[i])) {
                parameters_[i] = 0.0;
            }
        }
    }
    
    ModelState get_state() const {
        ModelState state;
        state.parameters = parameters_;
        state.rng_state = rng_->get_state();
        return state;
    }
    
    void set_state(const ModelState& state) {
        if (state.parameters.size() != parameters_.size()) {
            throw std::invalid_argument("Parameter count mismatch in set_state");
        }
        parameters_ = state.parameters;
        rng_->set_state(state.rng_state);
    }
    
    bool save_checkpoint(const std::string& filename) const {
        std::ofstream file(filename, std::ios::binary);
        if (!file) return false;
        
        ModelState state = get_state();
        size_t param_count = state.parameters.size();
        file.write(reinterpret_cast<const char*>(&param_count), sizeof(param_count));
        file.write(reinterpret_cast<const char*>(state.parameters.data()), 
                  param_count * sizeof(double));
        
        size_t state_size = state.rng_state.size();
        file.write(reinterpret_cast<const char*>(&state_size), sizeof(state_size));
        file.write(state.rng_state.data(), state_size);
        
        return file.good();
    }
    
    bool load_checkpoint(const std::string& filename) {
        std::ifstream file(filename, std::ios::binary);
        if (!file) return false;
        
        size_t param_count;
        file.read(reinterpret_cast<char*>(&param_count), sizeof(param_count));
        
        if (param_count != parameters_.size()) {
            return false;
        }
        
        ModelState state;
        state.parameters.resize(param_count);
        file.read(reinterpret_cast<char*>(state.parameters.data()), 
                 param_count * sizeof(double));
        
        size_t state_size;
        file.read(reinterpret_cast<char*>(&state_size), sizeof(state_size));
        
        state.rng_state.resize(state_size);
        file.read(&state.rng_state[0], state_size);
        
        set_state(state);
        return file.good();
    }
    
    const std::vector<double>& parameters() const { return parameters_; }
    size_t num_parameters() const { return parameters_.size(); }
    size_t num_qubits() const { return circuit_->num_qubits(); }
    const Observable& observable() const { return *observable_; }
};

class TrainingController {
    struct TrainingState {
        size_t iteration;
        double loss;
        double gradient_norm;
        QuantumModel::ModelState model_state;
        std::chrono::steady_clock::time_point timestamp;
        
        bool operator>(const TrainingState& other) const {
            return loss > other.loss;
        }
        
        bool operator<(const TrainingState& other) const {
            return loss < other.loss;
        }
    };
    
    struct TrainingStateComparator {
        bool operator()(const TrainingState& a, const TrainingState& b) const {
            return a.loss > b.loss;
        }
    };
    
    std::unique_ptr<QuantumModel> model_;
    std::vector<TrainingState> history_;
    std::priority_queue<TrainingState, 
                       std::vector<TrainingState>,
                       TrainingStateComparator> best_states_;
    std::atomic<bool> training_active_{false};
    std::string checkpoint_dir_;
    size_t checkpoint_interval_;
    size_t max_history_size_;
    
    TrainingState create_state(size_t iteration, double loss, 
                              const std::vector<double>& gradients) const {
        TrainingState state;
        state.iteration = iteration;
        state.loss = loss;
        state.model_state = model_->get_state();
        state.timestamp = std::chrono::steady_clock::now();
        
        double norm = 0.0;
        for (double g : gradients) {
            norm += g * g;
        }
        state.gradient_norm = std::sqrt(norm);
        
        return state;
    }
    
    void save_checkpoint_impl(size_t iteration) const {
        if (checkpoint_dir_.empty()) return;
        
        std::filesystem::create_directories(checkpoint_dir_);
        std::string filename = checkpoint_dir_ + "/checkpoint_" + 
                              std::to_string(iteration) + ".qckpt";
        
        if (!model_->save_checkpoint(filename)) {
            std::cerr << "Failed to save checkpoint at iteration " 
                      << iteration << std::endl;
        }
    }
    
    bool rollback_to_best() {
        if (best_states_.empty()) return false;
        
        TrainingState best = best_states_.top();
        model_->set_state(best.model_state);
        
        std::cout << "Rolled back to iteration " << best.iteration 
                  << " with loss " << best.loss << std::endl;
        
        return true;
    }
    
public:
    TrainingController(std::unique_ptr<QuantumModel> model,
                      const std::string& checkpoint_dir = "",
                      size_t checkpoint_interval = 100,
                      size_t max_history = 1000)
        : model_(std::move(model)), checkpoint_dir_(checkpoint_dir),
          checkpoint_interval_(checkpoint_interval), max_history_size_(max_history) {
        
        if (!checkpoint_dir_.empty()) {
            std::filesystem::create_directories(checkpoint_dir_);
        }
    }
    
    struct TrainingResult {
        bool success;
        size_t iterations;
        double final_loss;
        std::vector<TrainingState> history;
        std::exception_ptr error;
    };
    
    TrainingResult train(size_t max_iterations, double learning_rate,
                        double convergence_threshold = 1e-6,
                        size_t patience = 100) {
        training_active_ = true;
        
        TrainingResult result;
        size_t no_improvement_count = 0;
        double best_loss = std::numeric_limits<double>::max();
        
        try {
            for (size_t iter = 0; iter < max_iterations && training_active_; ++iter) {
                auto gradients = model_->compute_gradients();
                double current_loss = model_->loss();
                
                TrainingState state = create_state(iter, current_loss, gradients);
                history_.push_back(state);
                best_states_.push(state);
                
                if (current_loss < best_loss - convergence_threshold) {
                    best_loss = current_loss;
                    no_improvement_count = 0;
                } else {
                    no_improvement_count++;
                }
                
                if (no_improvement_count >= patience) {
                    std::cout << "Early stopping at iteration " << iter 
                              << " (no improvement for " << patience 
                              << " iterations)" << std::endl;
                    break;
                }
                
                model_->update_parameters(gradients, learning_rate);
                
                if (iter % checkpoint_interval_ == 0) {
                    save_checkpoint_impl(iter);
                }
                
                if (current_loss < convergence_threshold) {
                    std::cout << "Converged at iteration " << iter 
                              << " with loss " << current_loss << std::endl;
                    break;
                }
                
                if (history_.size() > max_history_size_) {
                    history_.erase(history_.begin());
                }
            }
            
            result.success = true;
            result.iterations = history_.size();
            result.final_loss = history_.empty() ? 0.0 : history_.back().loss;
            result.history = history_;
            
        } catch (const std::exception& e) {
            result.success = false;
            result.error = std::current_exception();
            
            std::cerr << "Training failed: " << e.what() << std::endl;
            
            if (!rollback_to_best()) {
                std::cerr << "Failed to rollback to previous state" << std::endl;
            }
        }
        
        training_active_ = false;
        return result;
    }
    
    void stop_training() {
        training_active_ = false;
    }
    
    bool resume_training(const std::string& checkpoint_file) {
        if (!model_->load_checkpoint(checkpoint_file)) {
            return false;
        }
        
        training_active_ = true;
        return true;
    }
    
    const std::vector<TrainingState>& get_history() const { return history_; }
    bool is_training() const { return training_active_; }
    
    std::optional<TrainingState> get_best_state() const {
        if (best_states_.empty()) return std::nullopt;
        return best_states_.top();
    }
};

} // namespace quantum
