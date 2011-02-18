#include "user_defined_types.h"

/// Kernel tabulating the element matrix for a 1D Laplace
/// parameters: A output element matrix (linearised)
///             x coordinates of element vertices
template < typename T >
inline void laplace(Data1D<T>& A, const Data1D<T>& x) {
  const T hinv = 1./(x[1] - x[0]);
  A[0] = hinv;
  A[1] = -hinv;
  A[2] = -hinv;
  A[3] = hinv;
}

