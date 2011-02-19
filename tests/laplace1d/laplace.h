#include "user_defined_types.h"

/// Kernel applying a 1D Laplace operator to a vector w to obtain a vector v
/// parameters: v output vector (over the element)
///             w input vector (over the element)
///             x coordinates of element vertices
template < typename T >
inline void laplace(Data1D<T>& v, const Data1D<T>& w, const Data1D<T>& x) {
  const T hinv = 1./(x(1) - x(0));
  v(0) = hinv*(w(0)-w(1));
  v(1) = hinv*(w(1)-w(0));
}

