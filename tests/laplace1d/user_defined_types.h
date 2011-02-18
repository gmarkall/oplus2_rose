/// Scalar field accessed through a mapping
template < typename T >
struct Data0D
{
  const T& operator()() const {
    return data[offset];
  }
  T& operator()() {
    return data[offset];
  }
  T* data;
  uint offset;
};

/// Vector field accessed through a mapping
template < typename T >
struct Data1D
{
  const T& operator()(uint i) const {
    return data[offset + stride * i];
  }
  T& operator()(uint i) {
    return data[offset + stride * i];
  }
  T* data;
  uint offset;
  uint stride;
};

/// 2nd-order tensor field accessed through a mapping
template < typename T >
struct Data2D
{
  const T& operator()(uint i, uint j) const {
    return data[offset[0] + stride[0] * (i + offset[1] + stride[1] * j)];
  }
  T& operator()(uint i, uint j) {
    return data[offset[0] + stride[0] * (i + offset[1] + stride[1] * j)];
  }
  T* data;
  uint offset[2];
  uint stride[2];
};

