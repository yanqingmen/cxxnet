#ifndef CXXNET_LAYER_OP_H_
#define CXXNET_LAYER_OP_H_
#pragma once
/*!
 * \file op.h
 * \brief extra mshadow operation for cxxnet
 * \author Bing Xu
 */
#include <mshadow/tensor.h>

namespace cxxnet {
/*! \brief operations for ActivationLayer */
namespace op {
/*! \brief Rectified Linear Operation */
struct identity {
  MSHADOW_XINLINE static real_t Map(real_t a) {
    return a;
  }
};
struct identity_grad {
  MSHADOW_XINLINE static real_t Map(real_t a) {
    return 1.0f;
  }
};

/*! \brief sigmoid unit */
struct sigmoid {
  MSHADOW_XINLINE static real_t Map(real_t a) {
    return 1.0f / (1.0f + expf(-a));
  }
};
struct sigmoid_grad {
  MSHADOW_XINLINE static real_t Map(real_t a) {
    return a * (1.0f - a);
  }
};
/*! \brief Rectified Linear Operation */
struct relu {
  MSHADOW_XINLINE static real_t Map(real_t a) {
    using namespace std;
    return max(a, 0.0f);
  }
};
struct relu_grad {
  MSHADOW_XINLINE static real_t Map(real_t a) {
    return a > 0.0f ? 1.0f : 0.0f;
  }
};

/*! \brief Leaky ReLU Operation */
struct xelu {
  MSHADOW_XINLINE static real_t Map(real_t a, real_t b) {
    return a > 0 ? a : a / b;
  }
};

struct xelu_grad {
  MSHADOW_XINLINE static real_t Map(real_t a, real_t b) {
    return a > 0 ? 1 : 1.0f / b;
  }
};

struct tanh {
  MSHADOW_XINLINE static real_t Map(real_t a) {
    return tanhf( a );
  }
};

struct tanh_grad {
  MSHADOW_XINLINE static real_t Map(real_t a) {
    return 1.0f - a * a;
  }
};


struct square {
  MSHADOW_XINLINE static real_t Map(real_t a) {
    return a * a;
  }
};

/*! \brief used for generate Bernoulli mask */
struct threshold {
  MSHADOW_XINLINE static real_t Map(real_t a, real_t b) {
    return a < b ? 1.0f : 0.0f;
  }
};

/*! \brief used for generate element of power */
struct power {
  MSHADOW_XINLINE static real_t Map(real_t a, real_t b) {
    return powf( a, b );
  }
};

/*!\ \brief used for generate element sqrt */
struct square_root {
  MSHADOW_XINLINE static real_t Map(real_t a) {
    return sqrt(a);
  }
};

/*! \brief used for gradient clipping and nan detection */
struct clip {
  MSHADOW_XINLINE static real_t Map(real_t a, real_t b) {
    if (isnan(a)) return 0.0f;
    if (a < -b) return -b;
    if (a > b) return b;
    return a;
  }
};

}  // namespace op
}  // namespace cxxnet
#endif // CXXNET_LAYER_OP_H
