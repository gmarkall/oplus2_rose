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

template < typename T >
class Coordinates
{
public:
  T& operator[](uint i);
};

template < typename T >
inline void laplace(Data1D<T>& A, const Data1D<T>& x) {
  const T hinv = 1./(x[1] - x[0]);
  A[0] = hinv;
  A[1] = -hinv;
  A[2] = -hinv;
  A[3] = hinv;
}

