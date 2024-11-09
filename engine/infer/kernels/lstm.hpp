#ifndef POCKET_AI_ENGINE_INFERENCE_KERNELS_LSTM_HPP_
#define POCKET_AI_ENGINE_INFERENCE_KERNELS_LSTM_HPP_

#include <stdint.h>
#include <algorithm>
#include <math.h>

#include "engine/infer/common.hpp"
#include "engine/infer/types.hpp"

#include "engine/infer/kernels/fully_connected.hpp"

namespace pai {
namespace infer {

// tensorflow\lite\micro\kernels\lstm_shared.h
// tensorflow\lite\micro\kernels\unidirectional_sequence_lstm.cc
// tensorflow\lite\micro\kernels\lstm_eval.cc
// Input Tensors of size {n_batch, n_input}
constexpr int kLstmInputTensor = 0;

// Input weight tensors of size: {n_cell, n_input}
constexpr int kLstmInputToInputWeightsTensor = 1;  // Optional
constexpr int kLstmInputToForgetWeightsTensor = 2;
constexpr int kLstmInputToCellWeightsTensor = 3;
constexpr int kLstmInputToOutputWeightsTensor = 4;

// Recurrent weight tensors of size {n_cell, n_output}
constexpr int kLstmRecurrentToInputWeightsTensor = 5;  // Optional
constexpr int kLstmRecurrentToForgetWeightsTensor = 6;
constexpr int kLstmRecurrentToCellWeightsTensor = 7;
constexpr int kLstmRecurrentToOutputWeightsTensor = 8;

// Peephole weights tensors of size {n_cell}, representing a diagonal matrix.
constexpr int kLstmCellToInputWeightsTensor = 9;    // Optional
constexpr int kLstmCellToForgetWeightsTensor = 10;  // Optional
constexpr int kLstmCellToOutputWeightsTensor = 11;  // Optional

// Gates bias tensors of size {n_cell}
constexpr int kLstmInputGateBiasTensor = 12;  // Optional
constexpr int kLstmForgetGateBiasTensor = 13;
constexpr int kLstmCellGateBiasTensor = 14;
constexpr int kLstmOutputGateBiasTensor = 15;

// Projection weight tensor of size {n_output, n_cell}
constexpr int kLstmProjectionWeightsTensor = 16;  // Optional
// Projection bias tensor of size {n_output}
constexpr int kLstmProjectionBiasTensor = 17;  // Optional

// These state tensors are defined as variable tensors, and will be modified by
// this op.
constexpr int kLstmOutputStateTensor = 18;
constexpr int kLstmCellStateTensor = 19;

// Layer norm coefficient tensors of size {n_cell}, representing a diagonal
// matrix.
constexpr int kLstmInputLayerNormCoefficientsTensor = 20;   // Optional
constexpr int kLstmForgetLayerNormCoefficientsTensor = 21;  // Optional
constexpr int kLstmCellLayerNormCoefficientsTensor = 22;    // Optional
constexpr int kLstmOutputLayerNormCoefficientsTensor = 23;  // Optional

// // Provide an interface to access the internal tensors and buffers used for LSTM
// // invocation. Constructed during the invocation phase
// struct LSTMKernelContents {
//     // Internal tensors, fixed (const). see lstm_shared.h for tensor names
//     const Tensor* GetInternalTensor(const int tensor_index) const {
//         return internal_tensors[tensor_index];
//     }
//     // Variable tensors (will be changed, can not be const)
//     Tensor* HiddenStateTensor() {
//         return internal_tensors[kLstmOutputStateTensor];
//     }
//     Tensor* CellStateTensor() {
//         return internal_tensors[kLstmCellStateTensor];
//     }
//     // Node internal tensors with indexes defined at the beginning of the file
//     Tensor* internal_tensors[24];
//     Tensor* output_tensor;
// };

struct LSTMBuffers {
    // TFLM buffers requires buffer index from LstmOpData.
    float* buffer0;
    float* buffer1;
    float* buffer2;
    float* buffer3;
};

// Size information about the LSTM kernel, which is deduced from tensors stored
// in the flat buffer file.
struct LstmSizeInfo {
    bool time_major;
    int batch_size;
    int time_steps;
    int input_dimension;
    int state_dimension;
};

// Contains information about the cell state tensor
struct CellStateInfo {
    float cell_clip;
    // clipping range for cell state only 16 bits cell is supported (could be
    // generalized through templatation)
    int16_t quantized_cell_clip;
    // 2^-cell_state_scale_power = cell state scale, required by integer tanh
    // computation
    int32_t cell_state_scale_power;
};

// Parameters for the two fully conncted computation inside each gate
struct GateParameters {
    // FullyConnectedParams input_fc_params;
    float input_fc_activation_min;
    float input_fc_activation_max;
    // FullyConnectedParams recurrent_fc_params;
    float recurrent_fc_activation_min;
    float recurrent_fc_activation_max;
};

typedef struct {
    float float_activation_min;
    float float_activation_max;
} ArithmeticParams;

// Paramaters for the element wise multiplications between gate outputs
struct InterGateParameters {
    ArithmeticParams forget_cell_mul_params;
    ArithmeticParams input_mul_params;
    ArithmeticParams output_mul_params;
};

// Possible fused activation functions.
typedef enum {
  kTfLiteActNone = 0,
  kTfLiteActRelu,
  kTfLiteActReluN1To1,  // min(max(-1, x), 1)
  kTfLiteActRelu6,      // min(max(0, x), 6)
  kTfLiteActTanh,
  kTfLiteActSignBit,
  kTfLiteActSigmoid,
} TfLiteFusedActivation;

// Contains required computation information for LSTM kernel evaluation.
// Specifically, it includes shape and quantization settings for the LSTM
// internal operations. Formatted to support operations defined in the
// tensorflow/lite/kernels/internal/reference/integer_ops
// Should be constructed during the preparation phase
struct OpDataLSTM {
    LstmSizeInfo size_info;
    CellStateInfo cell_state_info;
    TfLiteFusedActivation cell_gate_nonlinear_type;
    GateParameters forget_gate_parameters;
    GateParameters input_gate_parameters;
    GateParameters cell_gate_parameters;
    GateParameters output_gate_parameters;
    InterGateParameters inter_gate_parameters;
    void *buffer[4];  // ScratchBuffer -> LSTMBuffers buffers
};

typedef struct {
    uint32_t op_id;
    Tensor *input_tensor[24];   // in LSTMKernelContents
    Tensor *output_tensor;

    OpDataLSTM op_data;
} LstmParams;

// Manages the slice position (offset), slice length (sliced tensor shape),
// and update rules for input/output/hidden state/cell state tensors at each
// time step.
class LstmStepManager {
public:
    LstmStepManager() = delete;
    // Does not take any ownership, and all pointers must refer to valid objects
    // that outlive the one constructed.
    explicit LstmStepManager(const LstmSizeInfo* size_info)
        : size_info_(*size_info) {}

    // Increment the data offset so the sigle time step invocation call can access
    // the corresponding input/output tensor data at the time step  
    void UpdateTime() {
        current_time_ += 1;
        PAI_DCHECK_LE(current_time_, size_info_.time_steps);
        // default as one batch per inference
        int input_step = size_info_.input_dimension;
        int output_step = size_info_.state_dimension;
        // time major: batch inference
        if (size_info_.time_major) {
            input_step = input_step * size_info_.batch_size;
            output_step = output_step * size_info_.batch_size;
        }

        input_offset_ += input_step;
        output_offset_ += output_step;
    }

    // Increment the data offset so the sigle time step invocation call can access
    // the corresponding hidden/cell state tensor data at the time step (for single
    // batch inference only)
    void UpdateBatch() {
        current_batch_ += 1;
        PAI_DCHECK_LE(current_batch_, size_info_.batch_size);
        // batch inference for time major: no action needed
        if (size_info_.time_major) {
            return;
        }
        // otherwise: singe batch inference, go to the next batch
        hidden_state_offset_ += size_info_.state_dimension;
        cell_state_offset_ += size_info_.state_dimension;
    }
    void ResetTime() { current_time_ = 0; }
    // Input shape for each single time LSTM invocation.
    // Multi-batch for time_major input
    Shape InputShape() const {
        int batch_size = 1;
        if (size_info_.time_major) {
            batch_size = size_info_.batch_size;
        }
        Shape s;
        s.dims_count = 2;
        s.dims[0] = batch_size;
        s.dims[1] = size_info_.input_dimension;
        return s;
    }
    
    // State shape (both hidden and cell) for each single time LSTM invocation.
    // Multi-batch for time_major input
    Shape StateShape()  const {
        int batch_size = 1;
        if (size_info_.time_major) {
            batch_size = size_info_.batch_size;
        }
        Shape s;
        s.dims_count = 2;
        s.dims[0] = batch_size;
        s.dims[1] = size_info_.state_dimension;
        return s;
    }

    int InputOffset() const { return input_offset_; }
    int OutputOffset() const { return output_offset_; }
    int HiddenStateOffset() const { return hidden_state_offset_; }
    int CellStateOffset() const { return cell_state_offset_; }

private:
    int current_time_ = 0;
    int current_batch_ = 0;
    int input_offset_ = 0;
    int output_offset_ = 0;
    int hidden_state_offset_ = 0;
    int cell_state_offset_ = 0;
    // Sizeinfo is from LstmOpData, which reside in the memory arena
    // (guarante to outlast LSTMStepManager, which reside in stack)
    const LstmSizeInfo& size_info_;
};

void AddElementWise(const float* input_1, const float* input_2, int n_batch,
                    int n_input, float* output) {
    for (int batch = 0; batch < n_batch; ++batch) {
        for (int i = 0; i < n_input; ++i) {
            const int index = batch * n_input + i;
            output[index] = input_1[index] + input_2[index];
        }
    }
}

inline void Logistic(const Shape& input_shape, const float* input_data,
                     const Shape& output_shape, float* output_data) {
    const float cutoff_upper = 16.619047164916992188f;
    const float cutoff_lower = -9.f;

    const int flat_size = MatchingFlatSize(input_shape, output_shape);

    // Rational for using approximation in reference kernel.
    // 0. This approximation gives enough precision for float.
    // 1. This works around an issue on an embedded chipset where exp() does not
    // return correctly as expected - exp(x) should return inf when overflown
    // not 1.701417   IEEE 754 defines representation for inf.
    // 2. This will speed up calculation and is matching the behavior in the
    // optimized kernels. (check the definition of scalar_logistic_op<float>)

    for (int i = 0; i < flat_size; i++) {
        float val = input_data[i];
        float result;
        if (val > cutoff_upper) {
            result = 1.0f;
        } else if (val < cutoff_lower) {
            result = std::exp(val);
        } else {
            result = 1.f / (1.f + std::exp(-val));
        }
        output_data[i] = result;
    }
}

void Sigmoid(const Shape& data_shape, float* data) {
    Logistic(data_shape, data, data_shape, data);
}

inline void Tanh(const Shape& input_shape, const float* input_data,
                 const Shape& output_shape, float* output_data) {
    const int flat_size = MatchingFlatSize(input_shape, output_shape);

    for (int i = 0; i < flat_size; i++) {
        float val = input_data[i];
        float result = std::tanh(val);
        output_data[i] = result;
    }
}

void Tanh(int32_t cell_state_scale_power, const Shape& input_data_shape,
          float* input_data, const Shape& output_data_shape,
          float* output_data) {
    Tanh(input_data_shape, input_data, output_data_shape, output_data);
}

inline void Mul(const ArithmeticParams& params,
                const Shape& input1_shape, const float* input1_data,
                const Shape& input2_shape, const float* input2_data,
                const Shape& output_shape, float* output_data) {
    float output_activation_min = params.float_activation_min;
    float output_activation_max = params.float_activation_max;

    int flat_size = GetShapeFlatSize(output_shape);
    for (int i = 0; i < flat_size; ++i) {
        output_data[i] = ActivationFunctionWithMinMax(
            input1_data[i] * input2_data[i], output_activation_min,
            output_activation_max);
    }
}

// Input and output have the same shape in LSTM
void Mul(const Shape& shape, const ArithmeticParams& params,
         const float* input1_data, const float* input2_data,
         float* output_data) {
  return Mul(params, shape, input1_data, shape, input2_data,
                            shape, output_data);
}
// Calculates a single LSTM gate.
// Implements the following formula:
//   gate = activate(FC(input) + FC(recurrent))
// Activation is sigmoid except for the "cell" gate (configurable, usually tanh)
void CalculateLstmGate(
    const LstmStepManager& step_info, const GateParameters& gate_params,
    // Input FC
    const Tensor* input, Tensor* input_weight,
    Tensor* input_bias,
    // Recurrent FC
    const Tensor* recurrent, const Tensor* recurrent_weight,
    const Tensor* recurrent_bias,
    // Output
    float* gate_output,
    // Scratch arrays
    float* fc_output_buffer, const TfLiteFusedActivation activation) {
    
    const auto gate_output_shape = step_info.StateShape();
    // Check offset validity to avoid memory overflow
    PAI_DCHECK_LE(step_info.InputOffset() + GetShapeFlatSize(step_info.InputShape()),
                  GetShapeFlatSize(input->shape));
    PAI_DCHECK_LE(step_info.HiddenStateOffset() + GetShapeFlatSize(step_info.StateShape()),
                  GetShapeFlatSize(recurrent->shape));

    Tensor input_tensor, output_tensor, bias_tensor;
    // Input FC
    // Tensor bias_tensor = *input_weight;
    FullyConnectedParams input_fc_params;
    input_fc_params.op_id = -1;
    input_fc_params.float_activation_min = gate_params.input_fc_activation_min;
    input_fc_params.float_activation_max = gate_params.input_fc_activation_max;
    input_tensor.type = kPaiInferFloat32;
    input_tensor.data = (float *)input->data + step_info.InputOffset();
    input_tensor.shape = step_info.InputShape();
    input_fc_params.input_tensor = &input_tensor;
    input_fc_params.filter_tensor = *input_weight;
    input_fc_params.bias_tensor = *input_bias;
    output_tensor.type = kPaiInferFloat32;
    output_tensor.data = gate_output;
    output_tensor.shape = gate_output_shape;
    input_fc_params.output_tensor = &output_tensor;
    FullyConnected(input_fc_params);
    // FullyConnected(gate_params.input_fc_params, step_info.InputShape(),
    //                 tflite::micro::GetTensorData<float>(input) +
    //                     step_info.InputOffset(),
    //                 micro::GetTensorShape(input_weight),
    //                 tflite::micro::GetTensorData<WeightType>(input_weight),
    //                 tflite::micro::GetTensorShape(input_bias),
    //                 tflite::micro::GetOptionalTensorData<BiasType>(input_bias),
    //                 gate_output_shape, gate_output);
    
    // Recurrent FC
    FullyConnectedParams recurrent_fc_params;
    recurrent_fc_params.op_id = -2;
    recurrent_fc_params.float_activation_min = gate_params.recurrent_fc_activation_min;
    recurrent_fc_params.float_activation_max = gate_params.recurrent_fc_activation_max;
    input_tensor.type = kPaiInferFloat32;
    input_tensor.data = (float *)recurrent + step_info.HiddenStateOffset();
    input_tensor.shape = step_info.StateShape();
    recurrent_fc_params.input_tensor = &input_tensor;
    recurrent_fc_params.filter_tensor = *recurrent_weight;
    if (recurrent_bias)
        recurrent_fc_params.bias_tensor = *recurrent_bias;
    else {
        bias_tensor.type = kPaiInferFloat32;
        bias_tensor.data = nullptr;
        recurrent_fc_params.bias_tensor = bias_tensor;
    }
    output_tensor.type = kPaiInferFloat32;
    output_tensor.data = fc_output_buffer;
    output_tensor.shape = gate_output_shape;
    recurrent_fc_params.output_tensor = &output_tensor;
    FullyConnected(recurrent_fc_params);
    // FullyConnected(gate_params.recurrent_fc_params, step_info.StateShape(),
    //                 tflite::micro::GetTensorData<float>(recurrent) +
    //                     step_info.HiddenStateOffset(),
    //                 tflite::micro::GetTensorShape(recurrent_weight),
    //                 tflite::micro::GetTensorData<WeightType>(recurrent_weight),
    //                 tflite::micro::GetTensorShape(recurrent_bias),
    //                 tflite::micro::GetOptionalTensorData<BiasType>(recurrent_bias),
    //                 gate_output_shape, fc_output_buffer);

    AddElementWise(gate_output, fc_output_buffer,
                    /*n_batch=*/gate_output_shape.dims[0],
                    /*n_state=*/gate_output_shape.dims[1], gate_output);
    // Apply activation
    switch (activation) {
        case kTfLiteActSigmoid:
            Sigmoid(gate_output_shape, gate_output);
            break;
        case kTfLiteActTanh: {
            // Set the scale power to -12 to avoid shift
            Tanh(/*cell_state_scale_power=*/-12, gate_output_shape, gate_output,
                gate_output_shape, gate_output);
            } break;
        // default:
        //     // Only Sigmoid or Tanh is used.
        //     TFLITE_ASSERT_FALSE;
    }
}

void Clipping(const int v_size, const CellStateInfo& cell_state_info,
              float* vector) {
    for (int i = 0; i < v_size; i++) {
        vector[i] = std::max(std::min(cell_state_info.cell_clip, vector[i]),
                            -cell_state_info.cell_clip);
    }
}

// Update the cell state using the output from the forget gate, input gate, and
// cell gate Formula: updated_cell_state = forget_gate_output*cell_state +
// input_gate_output * cell_gate_output, where * denotes element wise
// multiplication
void UpdateLstmCell(const LstmStepManager& step_info,
                    Tensor* cell_state,
                    // Gate outputs
                    float* forget_gate_output,
                    const float* input_gate_output,
                    const float* cell_gate_output,
                    // Mul parameters
                    const ArithmeticParams& forget_cell_mul_params,
                    const ArithmeticParams& input_mul_params,
                    const CellStateInfo& cell_state_info, float* buffer) {
    // Check offset validity to avoid memory overflow
    PAI_DCHECK_LE(
        step_info.CellStateOffset() + GetShapeFlatSize(step_info.StateShape()),
        GetShapeFlatSize(cell_state->shape));

    auto cell_state_shape = step_info.StateShape();
    // Forget Gate x Cell State
    Mul(cell_state_shape, forget_cell_mul_params, forget_gate_output,
        (float *)cell_state->data + step_info.CellStateOffset(),
        (float *)cell_state->data + step_info.CellStateOffset());
    // Input Gate x Cell Gate
    Mul(cell_state_shape, input_mul_params, input_gate_output, cell_gate_output,
        buffer);

    // Update the cell state
    AddElementWise((float*)cell_state->data + step_info.CellStateOffset(),
                    buffer,
                    /*n_batch=*/cell_state_shape.dims[0],
                    /*n_state=*/cell_state_shape.dims[1],
                    (float*)cell_state->data + step_info.CellStateOffset());

    if (cell_state_info.cell_clip > 0) {
            Clipping(GetShapeFlatSize(cell_state_shape), cell_state_info,
                    (float*)cell_state->data + step_info.CellStateOffset());
    }
}


// Update the hidden state of the LSTM kernel using the following formula:
// updated_hidden_state = Tanh(updated_cell_state) * output_gate_output, * means
// element wise multiplication
void UpdateLstmHidden(const LstmStepManager& step_info,
                      Tensor* cell_state,
                      Tensor* hidden_state,
                      const float* output_gate_output,
                      const ArithmeticParams& mul_params,
                      int32_t cell_state_scale_power, float* buffer) {
  // Check offset validity to avoid memory overflow
  PAI_DCHECK_LE(
      step_info.CellStateOffset() + GetShapeFlatSize(step_info.StateShape()),
      GetShapeFlatSize(cell_state->shape));
  PAI_DCHECK_LE(
      step_info.HiddenStateOffset() + GetShapeFlatSize(step_info.StateShape()), 
      GetShapeFlatSize(hidden_state->shape));

  auto cell_state_shape = step_info.StateShape();
  float* cell_state_data = (float *)cell_state->data + step_info.CellStateOffset();
  // Tanh(cell_state)
  Tanh(cell_state_scale_power, cell_state_shape, cell_state_data,
       cell_state_shape, buffer);
  // Update the hidden state
  Mul(cell_state_shape, mul_params, buffer, output_gate_output,
      (float*)hidden_state->data + step_info.HiddenStateOffset());
}

void LstmStep(const LstmStepManager& step_info, const LstmParams& params,
              const LSTMBuffers& buffers) {
    /*Step1: Calculate gate outputs to prepare cell state update*/
    float* gate_internal_buffer = buffers.buffer3;
    float* forget_gate_output = buffers.buffer0;
    CalculateLstmGate(
        step_info, params.op_data.forget_gate_parameters,
        // Input FC
        params.input_tensor[kLstmInputTensor],
        params.input_tensor[kLstmInputToForgetWeightsTensor],
        params.input_tensor[kLstmForgetGateBiasTensor],
        // Recurrent FC
        params.input_tensor[kLstmOutputStateTensor],
        params.input_tensor[kLstmRecurrentToForgetWeightsTensor],
        /*recurrent_bias*/ nullptr,
        // Output
        forget_gate_output,
        // Scratch arrays
        gate_internal_buffer, kTfLiteActSigmoid);

    // Input Gate calculation;
    float* input_gate_output = buffers.buffer1;
    CalculateLstmGate(
        step_info, params.op_data.input_gate_parameters,
        // Input FC
        params.input_tensor[kLstmInputTensor],
        params.input_tensor[kLstmInputToInputWeightsTensor],
        params.input_tensor[kLstmInputGateBiasTensor],
        // Recurrent FC
        params.input_tensor[kLstmOutputStateTensor],
        params.input_tensor[kLstmRecurrentToInputWeightsTensor],
        /*recurrent_bias*/ nullptr,
        // Output
        input_gate_output,
        // Scratch arrays
        gate_internal_buffer, kTfLiteActSigmoid);

    // Cell Gate calculation
    float* cell_gate_output = buffers.buffer2;
    CalculateLstmGate(
        step_info, params.op_data.cell_gate_parameters,
        // Input FC
        params.input_tensor[kLstmInputTensor],
        params.input_tensor[kLstmInputToCellWeightsTensor],
        params.input_tensor[kLstmCellGateBiasTensor],
        // Recurrent FC
        params.input_tensor[kLstmOutputStateTensor],
        params.input_tensor[kLstmRecurrentToCellWeightsTensor],
        /*recurrent_bias*/ nullptr,
        // Output
        cell_gate_output,
        // Scratch arrays
        gate_internal_buffer, params.op_data.cell_gate_nonlinear_type);

    /*Step2: update the cell state */
    const InterGateParameters& inter_gate_params = params.op_data.inter_gate_parameters;
    float* updated_input_buffer = buffers.buffer1;  // reuse buffer

    UpdateLstmCell(step_info, params.input_tensor[kLstmCellStateTensor],
                            forget_gate_output, input_gate_output,
                            cell_gate_output,
                            inter_gate_params.forget_cell_mul_params,
                            inter_gate_params.input_mul_params,
                            params.op_data.cell_state_info, updated_input_buffer);

    /*Step3: update the hidden state */
    float* output_gate_output = buffers.buffer1;  // reuse buffer
    CalculateLstmGate(
        step_info, params.op_data.output_gate_parameters,
        // Input FC
        params.input_tensor[kLstmInputTensor],
        params.input_tensor[kLstmInputToOutputWeightsTensor],
        params.input_tensor[kLstmOutputGateBiasTensor],
        // Recurrent FC
        params.input_tensor[kLstmOutputStateTensor],
        params.input_tensor[kLstmRecurrentToOutputWeightsTensor],
        /*recurrent_bias*/ nullptr,
        // Output
        output_gate_output,
        // Scratch arrays
        gate_internal_buffer, kTfLiteActSigmoid);

    float* tanh_activated_cell_buffer = buffers.buffer0;  // reuse buffer
    UpdateLstmHidden(
        step_info, params.input_tensor[kLstmCellStateTensor],
        params.input_tensor[kLstmOutputStateTensor], output_gate_output,
        inter_gate_params.output_mul_params,
        params.op_data.cell_state_info.cell_state_scale_power,
        tanh_activated_cell_buffer);

    /*Step4: copy the update the hidden state to output*/
    // Check offset validity to avoid memory overflow
    //   PAI_DCHECK_LE(
    //       step_info.OutputOffset() + step_info.StateShape().FlatSize(),
    //       tflite::micro::GetTensorShape(kernel_content.output_tensor).FlatSize());
    // record the output (from the updated hidden state)
    float* output_ptr = (float *)params.output_tensor->data;
    const Tensor* hidden_state = params.input_tensor[kLstmOutputStateTensor];
    memcpy(output_ptr + step_info.OutputOffset(),
                (float *)hidden_state->data + step_info.HiddenStateOffset(),
                GetShapeFlatSize(step_info.StateShape()) * sizeof(float));
}

// ref: tensorflow\lite\micro\kernels\unidirectional_sequence_lstm.cc
inline void Lstm(const LstmParams& params) {

    LSTMBuffers buffers;
    buffers.buffer0 = (float *)params.op_data.buffer[0];
    buffers.buffer1 = (float *)params.op_data.buffer[1];
    buffers.buffer2 = (float *)params.op_data.buffer[2];
    buffers.buffer3 = (float *)params.op_data.buffer[3];
    printf("buffer: %p, %p, %p, %p.\n", buffers.buffer0, buffers.buffer1, buffers.buffer2, buffers.buffer3);

    LstmStepManager step_info(&params.op_data.size_info);
    const auto& size_info = params.op_data.size_info;
    // time is the first dimention, enable batch computation
    if (size_info.time_major) {
        for (int t = 0; t < size_info.time_steps; t++) {
            LstmStep(step_info, params, buffers);
            // prepare for the next time step
            step_info.UpdateTime();
        }
    } else {
        // batch first, unable to size the input data. single batch inference
        for (int b = 0; b < size_info.batch_size; b++) {
            for (int t = 0; t < size_info.time_steps; t++) {
                LstmStep(step_info, params, buffers);
                // prepare for the next time step
                step_info.UpdateTime();
            }
            // prepare for the next batch
            step_info.UpdateBatch();
            step_info.ResetTime();
        }
    }
}

} // namespace infer
} // namespace pai

#endif // POCKET_AI_ENGINE_INFERENCE_KERNELS_FULLY_CONNECTED_HPP_