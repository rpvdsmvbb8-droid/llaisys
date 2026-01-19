#include "op.hpp"
#include <cstring>

namespace llaisys::ops {
void linear(tensor_t out, tensor_t in, tensor_t weight, tensor_t bias) {

    size_t N = in->shape()[0];
    size_t K = in->shape()[1];
    size_t M = weight->shape()[0];

    auto load = [&](const std::byte* base,
                    size_t offset,
                    llaisysDataType_t dtype) -> float {
        switch (dtype) {
            case LLAISYS_DTYPE_F32: {
                float v;
                std::memcpy(&v, base + offset, sizeof(v));
                return v;
            }
            case LLAISYS_DTYPE_F16: {
                fp16_t v;
                std::memcpy(&v, base + offset, sizeof(v));
                return utils::cast<float>(v);
            }
            case LLAISYS_DTYPE_BF16: {
                bf16_t v;
                std::memcpy(&v, base + offset, sizeof(v));
                return utils::cast<float>(v);
            }
            default:
                EXCEPTION_UNSUPPORTED_DATATYPE(dtype);
        }
    };

    auto store = [&](std::byte* base,
                     size_t offset,
                     llaisysDataType_t dtype,
                     float value) {
        switch (dtype) {
            case LLAISYS_DTYPE_F32: {
                float v = value;
                std::memcpy(base + offset, &v, sizeof(v));
                break;
            }
            case LLAISYS_DTYPE_F16: {
                fp16_t v = utils::cast<fp16_t>(value);
                std::memcpy(base + offset, &v, sizeof(v));
                break;
            }
            case LLAISYS_DTYPE_BF16: {
                bf16_t v = utils::cast<bf16_t>(value);
                std::memcpy(base + offset, &v, sizeof(v));
                break;
            }
            default:
                EXCEPTION_UNSUPPORTED_DATATYPE(dtype);
        }
    };

    size_t in_elem  = utils::dsize(in->dtype());
    size_t w_elem   = utils::dsize(weight->dtype());
    size_t out_elem = utils::dsize(out->dtype());
    size_t b_elem   = bias ? utils::dsize(bias->dtype()) : 0;

    for (size_t i = 0; i < N; ++i) {
        for (size_t j = 0; j < M; ++j) {
            float sum = 0.f;

            for (size_t k = 0; k < K; ++k) {
                size_t in_off =
                    (i * in->strides()[0] + k * in->strides()[1]) * in_elem;

                size_t w_off =
                    (j * weight->strides()[0] + k * weight->strides()[1]) * w_elem;

                float x = load(in->data(), in_off, in->dtype());
                float w = load(weight->data(), w_off, weight->dtype());

                sum += x * w;
            }

            if (bias) {
                size_t b_off = j * bias->strides()[0] * b_elem;
                float b = load(bias->data(), b_off, bias->dtype());
                sum += b;
            }

            size_t out_off =
                (i * out->strides()[0] + j * out->strides()[1]) * out_elem;

            store(out->data(), out_off, out->dtype(), sum);
        }
    }
}


} // namespace llaisys::ops
